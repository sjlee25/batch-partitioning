import tvm
from tvm import relay
from tvm.contrib import graph_runtime
from tvm.relay import testing
import numpy as np
import threading
from math import floor, ceil, log2
import copy
import getopt
from os import path,_exit
import pickle
import sys
import time

# Prints usage with arguments
def PrintHelp():
    print('** A simple test tool for divding batch sizes with performance of each device **')
    print('< Usage > python3 [file name] [options]')
    print('< Options >')
    print('  -b, --batch   : batch size')
    print('  -n, --network : network to inference')
    print('  -c, --cpu     : use CPU')
    print('  -i, --igpu    : use intel iGPU')
    print('  -g=#, --num_gpus=#        : number of GPUs to use')
    print('  --gi=, --gpu_index=i,j,.. : target GPU indicies to use')
    
class Environment:
    def __init__(self, net='vgg-16', batch_size=0, logging=False):
        self.devices = []
        self.batch_size = batch_size
        self.net = net
        self.opt_level = 3
        self.test_times = 1
        self.run_times = 1
        self.logging = logging

    def GetBatches(self):
        batches = []
        for dev in self.devices:
            batches.append(dev.batch_size)
        return batches

    def get_network(self, name, batch_size, dtype='float32'):
        """Get the symbol definition and random weight of a network

        Parameters
        ----------
        name: str
            The name of the network, can be 'resnet-18', 'resnet-50', 'vgg-16', 'inception_v3', 'mobilenet', ...
        batch_size: int
            batch size
        dtype: str
            Data type

        Returns
        -------
        net: relay.Module
            The relay function of network definition
        params: dict
            The random parameters for benchmark
        input_shape: tuple
            The shape of input tensor
        output_shape: tuple
            The shape of output tensor
        """
        input_shape = (batch_size, 3, 224, 224)
        output_shape = (batch_size, 1000)

        if name == 'mobilenet':
            net, params = testing.mobilenet.get_workload(batch_size=batch_size, dtype=dtype)
        elif name == 'inception_v3':
            input_shape = (batch_size, 3, 299, 299)
            net, params = testing.inception_v3.get_workload(batch_size=batch_size, dtype=dtype)
        elif "resnet" in name:
            n_layer = int(name.split('-')[1])
            net, params = testing.resnet.get_workload(num_layers=n_layer, batch_size=batch_size, dtype=dtype)
        elif "vgg" in name:
            n_layer = int(name.split('-')[1])
            net, params = testing.vgg.get_workload(num_layers=n_layer, batch_size=batch_size, dtype=dtype)
        elif "densenet" in name:
            n_layer = int(name.split('-')[1])
            net, params = testing.densenet.get_workload(densenet_size=n_layer, batch_size=batch_size, dtype=dtype)
        elif "squeezenet" in name:
            version = name.split("_v")[1]
            net, params = testing.squeezenet.get_workload(batch_size=batch_size, version=version, dtype=dtype)
        # elif name == 'mxnet':
        #     # an example for mxnet model
        #     from mxnet.gluon.model_zoo.vision import get_model
        #     block = get_model('resnet18_v1', pretrained=True)
        #     net, params = relay.frontend.from_mxnet(block, shape={'data': input_shape}, dtype=dtype)
        #     net = net["main"]
        #     net = relay.Function(net.params, relay.nn.softmax(net.body), None, net.type_params, net.attrs)
        #     net = relay.Module.from_expr(net)
        else:
            raise ValueError("Unsupported network: " + name)

        return net, params, input_shape, output_shape

class Device:
    def __init__(self, dev_type='', idx=0):
        self.dev_type = dev_type
        self.idx = idx
        self.GetDevInfo()

        self.batch_size = 0
        self.exec_time = 0.0
        self.predict_time = 0.0

        # For partitioning
        self.all_time = 0.0
        self.eval_time = 0.0
        self.trial = 0
        self.diff = 0.0

    def GetDevInfo(self):
        if self.dev_type == 'cpu':
            self.ctx = tvm.cpu(self.idx)
            self.target = 'llvm'
            dev_name = self.ctx.device_name
            if dev_name is None:
                dev_name = 'Intel(R) Core(TM) i7-9700K CPU @3.60GHz'
            self.name = '[CPU] ' + dev_name
        
        elif self.dev_type == 'igp':
            self.ctx = tvm.opencl(self.idx)
            self.target = tvm.target.intel_graphics()
            dev_name = self.ctx.device_name
            self.name = '[IGPU] ' + dev_name

        elif self.dev_type == 'gpu':
            cudaInit = tvm.get_global_func('cudaInit')
            cudaInit(self.idx)
            self.ctx = tvm.gpu(self.idx)
            self.target = 'cuda'
            dev_name = self.ctx.device_name
            self.name = '[GPU] ' + dev_name

        else:
            print('[Error] Unknown Device Type %s' % (self.dev_type))
            exit(1)

    def PrintResult(self):
        string = '%s %d = %7.2f ms' % (self.dev_type, self.idx, self.exec_time)
        if self.predict_time > 0:
            string += ' | %7.2f ms' % (self.predict_time)
        string += ' [%3d]' % (self.batch_size)
        print(string)

    def Run(self, repeat_time=1, mode=''):
        global env
        if self.batch_size == 0:
            print('%s %d = 0 batch' % (self.dev_type, self.idx))
            return 0.0
        
        net, params, input_shape, output_shape = env.get_network(name=env.net, batch_size=self.batch_size)
        print('try relay build..')
        with relay.build_config(opt_level=env.opt_level):
            graph, lib, params = relay.build(net, target=self.target, params=params)
        print('graph built!')

        try: module = graph_runtime.create(graph, lib, self.ctx)
        except Exception as e:
            PrintError(e, self.dev_type, self.idx)
            _exit(1)
        data = tvm.nd.array((np.random.uniform(size=input_shape)).astype('float32'))
        module.set_input('data', data, **params)
        timer = module.module.time_evaluator('run', self.ctx, number=1, repeat=repeat_time)

        self.exec_time = np.mean(np.array(timer().results)) * 1000
        if mode != 'test':
            self.PrintResult()
            # throughput = self.batch_size / self.exec_time * 1000

            if env.logging:
                # thrput_path = self.dev_type + '_thrput.txt'
                latency_path = env.net + '_' + self.dev_type + '_latency.txt'
                # acc_path = self.dev_type + '_acc.txt'
                # with open(thrput_path, 'a') as log_file:
                #     log_file.write('%f\n' % (throughput))
                with open(latency_path, 'a') as log_file:
                    log_file.write('%f\n' % (self.exec_time))
                # print('[log] (%d) throughput: %.3f  latency: %.3f' % (self.batch_size, throughput, self.exec_time))
                # with open(acc_path, 'a') as log_file:
                #     log_file.write('%f\n' % ((self.exec_time - self.predict_time) / self.exec_time * 100))

        return self.exec_time

class Partitioner:
    def __init__(self, env):
        self.env = env
        self.table_path = 'perf_table'
        self.test_batches_cpu = [1, 2, 3, 4, 7, 8, 9, 15, 16, 17]
        self.test_batches_gpu = [1, 2, 3, 4, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65, 100]

        self.perf_table = {} # { 'dev_name': { perf_table }, ... }
        self.benchmark_time = 0.0
        self.offload_trial = 1
        
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
                self.env.net in self.perf_table[dev.name]:
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
                result = dev.Run(self.env.test_times, 'test') / batch_size
                dev_dict[batch_size] = result
            self.perf_table[dev.name][self.env.net] = copy.deepcopy(dev_dict)
            print('%s (%s)\n%s' % (dev.name, self.env.net, dev_dict))
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
                print(dev.all_time)
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

        dev_perf = self.perf_table[dev.name][self.env.net]
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
        if i == len(test_batches):
            table_idx = len(test_batches) - 2
        
        xp = [test_batches[table_idx], test_batches[table_idx+1]]
        yp = [dev_perf[xp[0]], dev_perf[xp[1]]]
        
        return np.interp(batch_size, xp, yp) * batch_size

    def OffloadDev(self, offload_dev, base_dev, prev_time):
        if self.offload_trial > base_dev.batch_size:
            return
        
        dev_times = []
        for dev in self.env.devices:
            batch_size = dev.batch_size
            if dev == offload_dev:
                batch_size += self.offload_trial
            eval_time = self.EstimateDevTime(dev, batch_size)
            dev_times.append(eval_time)
            if dev == offload_dev:
                dev.eval_time = eval_time
        cur_time = max(dev_times)

        print(offload_dev.name, dev_times, '/', cur_time, prev_time)
        if cur_time <= prev_time:
            offload_dev.trial = self.offload_trial
            offload_dev.all_time = cur_time
            offload_dev.diff = cur_time - dev.eval_time

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
        while tolerate_cnt < 5:
            loop_time = time.time()
            
            self.offload_trial = pow(2, tolerate_cnt)
            for dev in self.env.devices:
                dev.trial = 0
                dev.diff = float('-inf')

            base_dev.batch_size -= self.offload_trial
            if len(self.env.devices) > 2:
                for dev in self.env.devices:
                    if dev == base_dev: continue
                    t = threading.Thread(target=self.OffloadDev, args=(dev, base_dev, prev_time))
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
            print(offload_dev.name, cur_time, prev_time)

            if offloaded_cnt > 0 and cur_time <= prev_time:
                base_dev.batch_size -= 1
                offload_dev.batch_size += 1
                prev_time = cur_time
                base_dev = self.FindDev('base_next')
                tolerate_cnt = 0
                print('base: %s  time: %.2f' % (base_dev.name, prev_time))
            else:
                tolerate_cnt += 1
            cnt += 1
            loop_time = time.time() - loop_time
            print("[%2d]" % (cnt), self.env.GetBatches(), "%.2f ms" % (loop_time * 1000))
            print()

            if base_dev.batch_size == 1:
                break

        for dev in self.env.devices:
            dev.predict_time = self.EstimateDevTime(dev, dev.batch_size)
        search_time = (time.time() - search_time) * 1000
        print('Partitioning finished in %.2f ms' % (search_time))

def PrintError(e, type, idx):
    print('\n[Error] Executing with %s %d failed' % (type, idx))
    print(e)
    _exit(1)

def main(argv):
    global env
    try: opts, args = getopt.getopt(argv, 'hn:b:cig:l',
        ['help', 'network=', 'batch=', 'cpu', 'igpu', 'num_gpus=', 'ng=', 'gpu_index=', 'gi=', 'log'])
    except getopt.GetoptError:
        PrintHelp()
        exit(1)

    batch_size = 0
    net = 'vgg-16'
    use_cpu = use_igp = use_gpu = logging = False
    gpus_str = ''
    for opt, arg in opts:
        if opt in ('-h', '--help'):
            PrintHelp()
            exit(0)
        elif opt in ('-n', '--network'):
            net = arg
        elif opt in ('-b', '--batch'):
            batch_size = int(arg)
        elif opt in ('-c', '--cpu'):
            use_cpu = True
        elif opt in ('-i', '--igpu'):
            use_igp = True
        elif opt in ('-g', '--ng', '--num_gpus'):
            use_gpu = True
            num_gpu = int(arg)
        elif opt in ('--gi', '--gpu_index'):
            use_gpu = True
            gpus_str = arg
        elif opt in ('-l', '--log'):
            logging = True

    if batch_size == 0:
        PrintHelp()
        exit(1)

    env = Environment(net=net, batch_size=batch_size, logging=logging)
    if use_cpu:
        env.devices.append(Device('cpu', 0))
    if use_igp:
        env.devices.append(Device('igp', 0))
    if use_gpu:
        if gpus_str == '':
            gpu_idxs = set(i for i in range(num_gpu))
        else:
            gpu_idxs = set(map(int, gpus_str.split(', ')))
        for i in gpu_idxs:
            env.devices.append(Device('gpu', i))

    if use_gpu and len(gpu_idxs) < 1:
        PrintHelp()
        exit(1)

    threads = []

    elapsed_time = time.time()
    div_time = 0.0
    
    if len(env.devices) == 1:
        env.devices[0].batch_size = batch_size

    else:
        div_time = time.time()
        partitioner = Partitioner(env)
        partitioner.StartPartition()
        div_time = time.time() - div_time - partitioner.benchmark_time

    for dev in env.devices:
        t = threading.Thread(target=dev.Run)
        threads.append(t)
        t.start()
    for t in threads:
        t.join()

    elapsed_time = time.time() - elapsed_time
    # print('\nPartitioned Result:', work_sizes)
    print('All elapsed time: %.2f sec' % (elapsed_time))
    print('Partitioning time: %.2f ms' % (div_time * 1000))

if __name__ == '__main__':
    env = None
    main(sys.argv[1:])
