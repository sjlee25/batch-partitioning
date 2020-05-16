import tvm
from tvm import relay
from tvm.contrib import graph_runtime
from tvm.relay import testing
import copy
from math import floor, ceil, log2
import numpy as np
from os import path, _exit
import pickle
import threading
import time

class Partitioner:
    def __init__(self, env):
        self.env = env
        self.table_path = 'perf_table'
        self.test_batches_cpu = [1, 2, 3, 4, 7, 8, 9, 15, 16, 17, 31, 32, 33]
        self.test_batches_gpu = [1, 2, 3, 4, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129]

        self.perf_table = {} # { 'dev_name': { perf_table }, ... }
        self.benchmark_time = 0.0
        self.offload_trial = 1
        self.tolerate_limit = 2
        
    def CheckPerfTable(self):
        dev_dict = {}
        new_devs = 0

        print('Checking Performance Table...')
        if path.exists(self.table_path):
            with open(self.table_path, 'rb') as table_file:
                self.perf_table = pickle.load(table_file)

        for dev in self.env.devices:
            test_batches = self.test_batches_gpu
            if dev.dev_type == 'cpu':
                test_batches = self.test_batches_cpu

            if dev.name in self.perf_table and \
                self.env.network in self.perf_table[dev.name]:
                    print(dev.name, 'exists in the table')
                    continue
            else:
                if dev.name not in self.perf_table:
                    self.perf_table[dev.name] = {}
                print('testing', dev.name, '...')

            bench_time = time.time()
            dev_dict.clear()

            for batch_size in test_batches:
                if batch_size == 0: continue
                dev.batch_size = batch_size

                build_time = time.time()
                net, params, input_shape, output_shape = \
                    self.env.get_network(name=self.env.network, batch_size=dev.batch_size)
                with relay.build_config(opt_level=self.env.opt_level):
                    graph, lib, params = relay.build(net, target=dev.target, params=params)
                build_time = time.time() - build_time
                print('<%s> build time: %.3f sec' % (dev.name, build_time))

                result = dev.Run(graph, lib, params, input_shape, 
                                self.env.test_times, 'test') / batch_size
                dev_dict[batch_size] = result
            self.perf_table[dev.name][self.env.network] = copy.deepcopy(dev_dict)
            print('%s (%s)\n%s' % (dev.name, self.env.network, dev_dict))
            new_devs += 1

            bench_time = time.time() - bench_time
            self.benchmark_time += bench_time

        if new_devs > 0:
            bench_time = time.time()
            with open(self.table_path, 'wb') as table_file:
                pickle.dump(self.perf_table, table_file)
            bench_time = time.time() - bench_time
            self.benchmark_time += bench_time

    def FindDev(self, attr):
        best_dev = None
        max_val = float('-inf')
        min_val = float('inf')
        
        for dev in self.env.devices:
            if attr == 'min_all': # Min Time
                if dev.all_time == float('inf'):
                    continue
                if dev.all_time <= min_val:
                    best_dev = dev
                    max_val = dev.all_time

            if attr == 'max_diff': # Max Value
                if dev.diff >= max_val:
                    best_dev = dev
                    max_val = dev.diff

            elif attr == 'base_init': # Min Time
                batch_size = self.env.batch_size
                dev_time = self.EstimateDevTime(dev, batch_size)
                if dev_time < min_val:
                    best_dev = dev
                    min_val = dev_time
            
            elif attr == 'base_next': # Min Time
                dev_time = dev.eval_time
                if dev_time < min_val:
                    best_dev = dev
                    min_val = dev_time

            else:
                print('[Error] Unknown Attribute %s in FindMax' % (attr))
                _exit(1)

        return best_dev

    def EstimateDevTime(self, dev, batch_size):
        if batch_size == 0:
            return 0

        dev_perf = self.perf_table[dev.name][self.env.network]
        test_batches = self.test_batches_gpu
        if dev.dev_type == 'cpu':
            test_batches = self.test_batches_cpu

        table_idx = 0
        for i in range(len(test_batches)):
            if test_batches[i] == batch_size:
                return dev_perf[batch_size] * batch_size
                break
            if batch_size < test_batches[i]:
                table_idx = i - 1
                break
        if i == len(test_batches)-1:
            table_idx = len(test_batches)-2
        
        xp = [test_batches[table_idx], test_batches[table_idx+1]]
        yp = [dev_perf[xp[0]], dev_perf[xp[1]]]
        
        return np.interp(batch_size, xp, yp) * batch_size

    def OffloadDev(self, offload_dev, base_dev, max_time):
        if self.offload_trial > base_dev.batch_size:
            return
        
        dev_times = []
        for dev in self.env.devices:
            if dev == base_dev: continue

            batch_size = dev.batch_size
            if dev == offload_dev:
                batch_size += self.offload_trial
            eval_time = self.EstimateDevTime(dev, batch_size)
            
            if dev == offload_dev:
                dev.eval_time = eval_time
            dev_times.append(eval_time)

        # print(offload_dev.name, 'prev:', prev_time)
        # print(dev_times)

        for dev_time in dev_times:
            if dev_time > max_time:
                return

        offload_dev.trial = self.offload_trial
        offload_dev.all_time = max_time
        offload_dev.diff = max_time - offload_dev.eval_time

    def StartPartition(self):
        self.CheckPerfTable()
        print('\nStart Partitioning...')

        for dev in self.env.devices:
            dev.batch_size = 0

        base_dev = self.FindDev('base_init')
        base_dev.batch_size = self.env.batch_size
        prev_time = self.EstimateDevTime(base_dev, base_dev.batch_size)

        cnt = 0
        offloaded_cnt = 1
        tolerate_cnt = 0
        threads = []

        search_time = time.time()
        while tolerate_cnt < self.tolerate_limit:
            loop_time = time.time()
            
            self.offload_trial = pow(2, tolerate_cnt)
            for dev in self.env.devices:
                dev.trial = 0
                dev.diff = float('-inf')

            base_dev.batch_size -= self.offload_trial
            max_time = self.EstimateDevTime(base_dev, base_dev.batch_size)
            if len(self.env.devices) > 2:
                for dev in self.env.devices:
                    if dev == base_dev: continue
                    t = threading.Thread(target=self.OffloadDev, args=(dev, base_dev, max_time))
                    threads.append(t)
                    t.start()
                for t in threads:
                    t.join()
            else: # threads are not needed
                for dev in self.env.devices:
                    if dev == base_dev: continue
                    self.OffloadDev(dev, base_dev, prev_time)
            base_dev.batch_size += self.offload_trial

            offload_dev = self.FindDev('max_diff')
            if offload_dev is None: break
            offloaded_cnt = offload_dev.trial
            cur_time = offload_dev.all_time

            if offloaded_cnt > 0 and cur_time <= prev_time:
                base_dev.batch_size -= 1
                offload_dev.batch_size += 1
                prev_time = cur_time
                base_dev = self.FindDev('base_next')
                tolerate_cnt = 0
                print('')
            else:
                tolerate_cnt += 1
            cnt += 1
            loop_time = time.time() - loop_time
            print("[%2d]" % (cnt), self.env.GetBatches(), "%.2f ms" % (loop_time * 1000))

            if base_dev.batch_size == 1:
                break

        for dev in self.env.devices:
            dev.predict_time = self.EstimateDevTime(dev, dev.batch_size)
        search_time = (time.time() - search_time) * 1000
        print('Partitioning finished in %.2f ms\n' % (search_time))
