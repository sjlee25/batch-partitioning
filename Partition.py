import tvm
from tvm import relay
from tvm.contrib import graph_runtime
from tvm.relay import testing
from util import get_network
import copy
from math import floor, ceil, log2
import numpy as np
from os import path, _exit
import pickle
import threading
import time

class PerfInfo:
    def __init__(self, exec_t, io_t):
        self.exec_time = exec_t
        self.io_time = io_t

    def getTimes(self):
        return self.exec_time, self.io_time

    def getSummedTime(self):
        return self.exec_time + self.io_time

class Partitioner:
    def __init__(self, env):
        self.env = env
        self.with_io_time = True
        self.table_path = 'table_' + env.network
        self.test_batches_gpu = [1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129]
        self.test_batches_cpu = self.test_batches_gpu[:-2] # max: 127
        self.perf_table = {}
        self.benchmark_time = 0.

        self.offload_trial = 1
        self.tolerate_limit = 1
        self.base_var_limit = 1
        self.test_limit = 3
        self.offload_thresh = 1.
        
    def checkPerfTable(self):
        print('Checking Performance Table...')
        if path.exists(self.table_path):
            with open(self.table_path, 'rb') as table_file:
                self.perf_table = pickle.load(table_file)

        for dev in self.env.devices:
            if dev.dev_type == 'gpu':
                test_batches = self.test_batches_gpu
            else:
                test_batches = self.test_batches_cpu

            if dev.name not in self.perf_table:
                self.perf_table[dev.name] = {}
                max_checked_batch = 0
                print('testing', dev.name, '...')
            else:
                print(dev.name, 'exists in the table')
                max_checked_batch = max(self.perf_table[dev.name].keys())

            if max_checked_batch >= test_batches[-1]:
                continue
            
            # start with the smallest batch size
            if max_checked_batch == 0: start_idx = 0
            else: start_idx = test_batches.index(max_checked_batch) + 1
            batch_size = dev.batch_size = test_batches[start_idx]
            print('  restart testing from batch size %d...' % (batch_size))

            net, params, input_shape, output_shape = \
                get_network(name=self.env.network, batch_size=batch_size)
            with relay.build_config(opt_level=self.env.opt_level):
                graph, lib, params = relay.build(net, target=dev.target, params=params)
            exec_time, io_time = dev.run(graph, lib, params, input_shape, self.env.test_times, 'test')

            if exec_time <= 0: break
            prev_time = exec_time
            prev_batch = batch_size
            self.perf_table[dev.name][batch_size] = PerfInfo(exec_time/batch_size, io_time/batch_size)

            for i in range(start_idx + 1, len(test_batches)):
                batch_size = dev.batch_size = test_batches[i]
                exec_time = float('inf')
                # batch_cnt = 0
                
                # build_time = time.time()
                net, params, input_shape, output_shape = \
                    get_network(name=self.env.network, batch_size=batch_size)
                with relay.build_config(opt_level=self.env.opt_level):
                    graph, lib, params = relay.build(net, target=dev.target, params=params)
                # build_time = time.time() - build_time
                # print('<%s> build time: %.3f sec' % (dev.name, build_time))

                # while exec_time > prev_time * (batch_size / prev_batch) and batch_cnt <= self.test_limit:
                #     exec_time, io_time = dev.run(graph, lib, params, input_shape, self.env.test_times, 'test')
                #     if exec_time <= 0: break
                #     batch_cnt += 1

                exec_time, io_time = dev.run(graph, lib, params, input_shape, self.env.test_times, 'test')
                if exec_time <= 0: break
                prev_time = exec_time
                prev_batch = batch_size
                self.perf_table[dev.name][batch_size] = PerfInfo(exec_time/batch_size, io_time/batch_size)

                with open(self.table_path, 'wb') as table_file:
                    pickle.dump(self.perf_table, table_file)

    def estimateDevTime(self, dev, batch_size):
        if batch_size == 0:
            return 0

        dev_perf = self.perf_table[dev.name]
        test_batches = self.test_batches_gpu
        if dev.dev_type == 'cpu':
            test_batches = self.test_batches_cpu

        if batch_size in dev_perf:
            return dev_perf[batch_size].exec_time * batch_size
        
        max_key = max(dev_perf.keys())
        if batch_size > max_key:
            return dev_perf[max_key].exec_time * batch_size

        min_val = 2**int(log2(batch_size))+1
        max_val = 2*(min_val-1)-1
        xp = [min_val, max_val]
        yp_exec = [dev_perf[xp[0]].exec_time, dev_perf[xp[1]].exec_time]

        intp_time = np.interp(batch_size, xp, yp_exec) * batch_size
        if self.with_io_time:
            yp_io = [dev_perf[xp[0]].io_time, dev_perf[xp[1]].io_time]
            intp_io_time = np.interp(batch_size, xp, yp_io) * batch_size
            intp_time += intp_io_time

        return intp_time

    def getMaxDiffDev(self):
        ret_dev = None
        max_val = float('-inf')
        for dev in self.env.devices:
            if dev.diff >= max_val:
                ret_dev = dev
                max_val = dev.diff
        return ret_dev

    def getInitBaseDev(self):
        ret_dev = None
        min_val = float('inf')
        for dev in self.env.devices:
            dev_time = self.estimateDevTime(dev, self.env.batch_size)
            if dev_time < min_val:
                ret_dev = dev
                min_val = dev_time
        return ret_dev

    def getNextBaseDev(self):
        ret_dev = None
        max_val = float('-inf')
        for dev in self.env.devices:
            dev_time = self.estimateDevTime(dev, self.env.batch_size)
            if dev_time > max_val:
                ret_dev = dev
                max_val = dev_time
        return ret_dev

    def offloadDev(self, offload_dev, base_dev, max_time):
        if self.offload_trial > base_dev.batch_size:
            return
        
        dev_times = []
        off_dev_time = 0.0
        for dev in self.env.devices:
            if dev == base_dev: continue

            batch_size = dev.batch_size
            if dev == offload_dev:
                batch_size += self.offload_trial
            eval_time = self.estimateDevTime(dev, batch_size)
            
            if dev == offload_dev:
                off_dev_time = eval_time
            dev_times.append(eval_time)

        for dev_time in dev_times:
            if dev_time > max_time * self.offload_thresh:
                return

        offload_dev.trial = self.offload_trial
        offload_dev.eval_time = off_dev_time
        offload_dev.diff = max_time - offload_dev.eval_time


    def startPartition(self):
        self.benchmark_time = time.time()
        self.checkPerfTable()
        self.benchmark_time = time.time() - self.benchmark_time

        print('\nStart Partitioning...')

        for dev in self.env.devices:
            dev.batch_size = 0

        base_dev = self.getInitBaseDev()
        base_dev.batch_size = self.env.batch_size

        cnt = 0
        offloaded_cnt = 1
        tolerate_cnt = base_var_cnt = 0
        threads = []

        search_time = time.time()
        while base_var_cnt < self.base_var_limit:
            loop_time = time.time()
            
            if cnt > 0 and tolerate_cnt > self.tolerate_limit:
                base_dev = self.getNextBaseDev()
                base_var_cnt += 1
                tolerate_cnt = 0

            max_time = self.estimateDevTime(base_dev, base_dev.batch_size - 1)

            self.offload_trial = 2**tolerate_cnt
            for dev in self.env.devices:
                dev.trial = 0
                dev.diff = float('-inf')

            if len(self.env.devices) > 2:
                for dev in self.env.devices:
                    if dev == base_dev: continue
                    t = threading.Thread(target=self.offloadDev, args=(dev, base_dev, max_time))
                    threads.append(t)
                    t.start()
                for t in threads:
                    t.join()
            else: # threads are not needed
                for dev in self.env.devices:
                    if dev == base_dev: continue
                    self.offloadDev(dev, base_dev, max_time)

            offload_dev = self.getMaxDiffDev()
            if offload_dev is None: break
            offloaded_cnt = offload_dev.trial

            if offloaded_cnt > 0:
                base_dev.batch_size -= offloaded_cnt
                offload_dev.batch_size += offloaded_cnt
                tolerate_cnt = 0
            else:
                tolerate_cnt += 1
            cnt += 1
            loop_time = time.time() - loop_time
            print("[%2d]" % (cnt), self.env.getBatches(), "%.2f ms" % (loop_time * 1000))

            if base_dev.batch_size == 1:
                break

        for dev in self.env.devices:
            dev.predict_time = self.estimateDevTime(dev, dev.batch_size)
        search_time = (time.time() - search_time) * 1000
        print('Partitioning finished in %.2f ms\n' % (search_time))
