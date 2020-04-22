import numpy as np
import tvm
from tvm import relay
from tvm.contrib import graph_runtime
from tvm.relay import testing
import threading
import time
from os import path, _exit
import copy
import pickle
from math import ceil, log2

def PrintError(e, name):
    print('\n[Error] Executing with %s failed' % (type, name))
    print(e)
    _exit(1)

class CandidateDevice:
    def __init__(self, trial_cnt, diff):
        self.trial_cnt = trial_cnt
        self.diff = diff
        self.exec_time = 0

class Partitioner:
    def __init__(self, batch_size, enable_devs, gpu_idxs):
        self.enable_devs = enable_devs # length-3 list
        self.dev_names = []
        self.num_devs = sum(enable_devs) # int
        self.gpu_start_idx = enable_devs[0] + enable_devs[1]
        self.gpu_idxs = gpu_idxs

        self.batch_size = batch_size
        self.max_steps = 1024
        self.granularity = ceil(batch_size / self.max_steps)

        self.file_name = 'perf_table'
        if path.exists(self.file_name):
            self.perf_table_file = open(self.file_name, 'rb')
        else: self.perf_table_file = None
        self.test_batches_cpu = [1, 2, 4, 8, 16]
        self.test_batches_gpu = [1, 2, 4, 8, 16, 32, 64, 128, 256]

        self.perf_table = {} # { 'dev_name': { perf_table }, ... }
        self.candidate_devs = []
        self.partition = []
        self.estimated_time = []
        self.benchmark_time = 0

        # Get device names
        if enable_devs[0] == 1:
            ctx = tvm.cpu(0)
            dev_name = ctx.device_name
            if dev_name is None:
                dev_name = 'Intel(R) Core(TM) i7-9700K CPU @ 3.60GHz'
                # dev_name = 'Intel(R) Core(TM) i7-8700K CPU @ 3.70GHz'
            self.dev_names.append('[CPU] ' + dev_name)

        if enable_devs[1] == 1:
            ctx = tvm.opencl(0)
            self.dev_names.append('[iGPU] ' + ctx.device_name)
            
        if enable_devs[2] > 0:
            # temporary codes for initializing GPU
            # image_shape = (3, 224, 224)
            # mod, params = relay.testing.vgg.get_workload(
            #     num_layers=16, batch_size=1, image_shape=image_shape)
            # with relay.build_config(opt_level=0):
            #     graph, lib, param = relay.build_module.build(mod, 'cuda', params=params)

            cudaInit = tvm.get_global_func('cudaInit')
            for i in self.gpu_idxs:
                cudaInit(i)
                ctx = tvm.gpu(i)
                self.dev_names.append('[GPU] ' + ctx.device_name)

    def GetDevInfo(self, dev_idx):
        if dev_idx < self.enable_devs[0]:
            ctx = tvm.cpu(0)
            target = 'llvm'
            
        elif dev_idx < self.gpu_start_idx:
            ctx = tvm.opencl(0)
            target = tvm.target.intel_graphics()
            
        else:
            ctx = tvm.gpu(dev_idx - self.gpu_start_idx)
            target = 'cuda'

        return ctx, target, self.dev_names[dev_idx]

    def RunDev(self, target, ctx, batch_size, dev_name):
        opt_level = 3
        num_class = 1000
        image_shape = (3, 224, 224)
        data_shape = (batch_size, ) + image_shape
        out_shape = (batch_size, num_class)

        data = np.random.uniform(-1, 1, size=data_shape).astype('float32')
        mod, params = relay.testing.vgg.get_workload(
            num_layers=16, batch_size=batch_size, image_shape=image_shape)
        with relay.build_config(opt_level=opt_level):
            graph, lib, param = relay.build_module.build(mod, target, params=params)

        try: dev = graph_runtime.create(graph, lib, ctx) 
        except Exception as e:
            PrintError(e, dev_name, '')
            batch_idx = self.test_batches.index(batch_size)
            return -1
        else:
            dev.set_input('data', data, **param)
            dev = dev.module
            dev = dev.time_evaluator('run', ctx, 1, 3)
            return np.mean(np.array(dev().results)) * 1000 / batch_size

    def CheckPerfTable(self):
        print('Checking Performance Table...')
        if self.perf_table_file is not None:
            self.perf_table = pickle.load(self.perf_table_file)
        dev_dict = {}
        new_devs = 0

        for i in range(self.num_devs):
            ctx, target, dev_name = self.GetDevInfo(i)

            if dev_name in self.perf_table:
                print(dev_name, 'exists in the table')
                continue
            else: print('testing', dev_name, '...')

            bench_time = time.time()
            dev_dict.clear()

            if i < self.enable_devs[0]:
                self.test_batches = self.test_batches_cpu
            else: self.test_batches = self.test_batches_gpu

            for bs in self.test_batches:
                if bs == 0: continue
                result = self.RunDev(target, ctx, bs, dev_name)
                if result < 0: break
                else: dev_dict[bs] = result
            self.perf_table[dev_name] = copy.deepcopy(dev_dict)
            print(dev_name, dev_dict)
            new_devs += 1

            bench_time = time.time() - bench_time
            self.benchmark_time += bench_time

        if new_devs > 0:
            bench_time = time.time()
            if self.perf_table_file is not None:
                self.perf_table_file.close()
            self.perf_table_file = open(self.file_name, 'wb')
            pickle.dump(self.perf_table, self.perf_table_file)

            bench_time = time.time() - bench_time
            self.benchmark_time += bench_time

        self.perf_table_file.close()

    def FindBaseDev(self):
        if self.enable_devs[2] > 0: # GPU
            if self.enable_devs[2] == 1: return self.gpu_start_idx
            else:
                # Find the fastest GPU
                gpu_times = []
                for i in range(self.enable_devs[2]):
                    gpu_times.append(self.EstimateDevTime(self.gpu_start_idx + i, self.batch_size))
                return gpu_times.index(min(gpu_times)) + self.gpu_start_idx

        elif self.enable_devs[1] > 0: return 1 # iGPU
        else: return 0 # CPU

    def FindMaxDiffIdx(self):
        diffs = []
        for c in self.candidate_devs:
            diffs.append(c.diff)
        
        max_idx = 0
        for i in range(len(diffs)):
            if diffs[i] >= diffs[max_idx]:
                max_idx = i

        return max_idx

    def EstimateDevTime(self, dev_idx, batch_size):
        if batch_size == 0:
            return 0

        test_batch_idx = 0
        _, _, dev_name = self.GetDevInfo(dev_idx)
        dev_perf = self.perf_table[dev_name]
        
        if dev_idx < self.gpu_start_idx:
            test_batches = self.test_batches_cpu
        else:
            test_batches = self.test_batches_gpu

        table_idx = int(log2(batch_size))
        if table_idx >= len(test_batches)-1:
            table_idx = len(test_batches)-2
        
        xp = [test_batches[table_idx], test_batches[table_idx+1]]
        yp = [dev_perf[xp[0]], dev_perf[xp[1]]]
        
        inf_time = np.interp(batch_size, xp, yp) * batch_size
        # io_time = 0.0
        # if dev_idx >= self.gpu_start_idx:
        #     io_time += 200.0

        return inf_time
        # return inf_time + io_time

    def EstimateAllTimes(self, partition):
        devs_time = []
        for i in range(self.num_devs):
            devs_time.append(self.EstimateDevTime(i, partition[i]))
        return devs_time

    def OffloadDev(self, dev_idx, base_dev, prev_time):
        offloading_trial = self.granularity
        if offloading_trial > self.partition[base_dev]:
            return
        
        partition = copy.deepcopy(self.partition)
        partition[base_dev] -= offloading_trial
        partition[dev_idx] += offloading_trial

        devs_time = self.EstimateAllTimes(partition)
        cur_time = max(devs_time)

        if cur_time < prev_time:
            self.candidate_devs[dev_idx].trial_cnt = offloading_trial
            self.candidate_devs[dev_idx].diff = prev_time - cur_time
            self.candidate_devs[dev_idx].exec_time = cur_time

    def StartPartition(self):
        self.CheckPerfTable()
        print('\nStart Partitioning...')

        for _ in range(self.num_devs):
            self.partition.append(0)
            self.candidate_devs.append(CandidateDevice(0, float('inf')))

        base_dev = self.FindBaseDev()
        self.partition[base_dev] = self.batch_size
        prev_time = self.EstimateDevTime(base_dev, self.partition[base_dev])

        cnt = 0
        offloaded_cnt = 1
        prev_partition = []
        offload_threads = []

        while offloaded_cnt > 0:
            loop_time = time.time()

            for i in range(self.num_devs):
                self.candidate_devs[i].trial_cnt = 0
                self.candidate_devs[i].diff = 0

            for i in range(self.num_devs):
                if i == base_dev: continue
                t = threading.Thread(target=self.OffloadDev, args=(i, base_dev, prev_time))
                offload_threads.append(t)
                t.start()
            for t in offload_threads:
                t.join()

            offload_dev = self.FindMaxDiffIdx()
            offloaded_cnt = self.candidate_devs[offload_dev].trial_cnt
            
            self.partition[offload_dev] += offloaded_cnt
            self.partition[base_dev] -= offloaded_cnt
            prev_time = self.candidate_devs[offload_dev].exec_time

            cnt += 1
            loop_time = time.time() - loop_time
            print("[%d]" % (cnt), self.partition, "%.2f ms" % (loop_time * 1000))

        self.estimated_time = self.EstimateAllTimes(self.partition)

if __name__ == "__main__":
    partitioner = Partitioner(100, [1, 1, 1])
    partitioner.StartPartition()