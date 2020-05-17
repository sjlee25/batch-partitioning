import tvm
from tvm import relay
from tvm.contrib import graph_runtime
from tvm.relay import testing
import argparse
import copy
import getopt
from math import floor, ceil, log2
import numpy as np
from os import path,_exit
import sys
import threading
import time
from Partition import Partitioner

import openpyxl # for excel logging
    
class Environment:
    def __init__(self, network, batch_size, devices, logging=False):
        self.devices = copy.deepcopy(devices)
        self.batch_size = batch_size
        self.network = network
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
        self.prev_bt = 0.0

    def GetDevInfo(self):
        if self.dev_type == 'cpu':
            self.ctx = tvm.cpu(self.idx)
            # self.target = 'llvm'
            self.target = 'llvm -mcpu=core-avx2' # reduces CPU time to half
            dev_name = self.ctx.device_name
            if dev_name is None:
                dev_name = 'Intel(R) Core(TM) i7-9700K CPU @3.60GHz'
                # dev_name = 'Intel(R) Core(TM) i7-8700K CPU @3.60GHz'
        
        elif self.dev_type == 'igpu':
            self.ctx = tvm.opencl(self.idx)
            self.target = tvm.target.intel_graphics()
            # self.target = tvm.target.intel_graphics(model='Intel® Processor Graphics Gen9')
            dev_name = self.ctx.device_name

        elif self.dev_type == 'gpu':
            cudaInit = tvm.get_global_func('cudaInit')
            cudaInit(self.idx)
            self.ctx = tvm.gpu(self.idx)
            # self.target = 'cuda'
            self.target = 'cuda -device=1050ti'
            dev_name = self.ctx.device_name

        else:
            print('[Error] Unknown Device Type %s' % (self.dev_type))
            exit(1)

        self.name = '[%s] '%(self.dev_type.upper()) + dev_name

    # Need to think about the race condition
    def PushResult(self):
        global result
        string = '%4s %d = %7.2f ms' % (self.dev_type.upper(), self.idx, self.exec_time)
        if self.predict_time > 0:
            string += ' | %7.2f ms' % (self.predict_time)
        string += ' [%3d]\n' % (self.batch_size)
        result += string

    def Run(self, graph, lib, params, input_shape, repeat_time=1, mode=''):
        global env, result
        if self.batch_size == 0:
            result += '%s %d = 0 batch\n' % (self.dev_type, self.idx)
            return 0.0

        try: module = graph_runtime.create(graph, lib, self.ctx)
        except Exception as e:
            dev_type = self.dev_type.upper()
            print('\n[Error] Executing with %s (%s %d) failed'
                % (self.name.replace('[%s] '%(dev_type), ''), dev_type, self.idx))
            if mode == 'test': return -1
            else: _exit(1)

        data = tvm.nd.array((np.random.uniform(size=input_shape)).astype('float32'))
        module.set_input('data', data, **params)
        timer = module.module.time_evaluator('run', self.ctx, number=1, repeat=repeat_time)

        self.exec_time = np.mean(np.array(timer().results)) * 1000
        if mode != 'test':
            self.PushResult()

            if env.logging:
                file_name = 'log.xlsx'
                if path.exists(file_name):
                    book = openpyxl.load_workbook(file_name)
                    if env.network in book:
                        sheet = book[env.network]
                    else: sheet = book.create_sheet(env.network)
                else:
                    book = openpyxl.Workbook()
                    sheet = book.create_sheet(env.network)
                for i in range(len(env.devices)):
                    sheet[str(chr(i + 65)) + str(batch_size)] = env.devices[i].batch_size
                book.save(file_name)

        return self.exec_time

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--network', type=str, choices=
                        ['mobilenet', 'resnet-18', 'resnet-34', 'resnet-50',
                        'vgg-16', 'vgg-19', 'inception_v3', 'densenet-121',
                        'squeezenet_v1.0', 'squeezenet_v1.1'], 
                        help='The name of neural network to inference')
    parser.add_argument('--batch', type=int, help='Batch size')
    parser.add_argument('--device', type=str, default='gpu0',
                        help='Devices to use, give as \'cpu\', \'igpu\' or \'gpu0\'')
    parser.add_argument('--log', type=str, choices=['enable', 'disable'], default='disable',
                        help='Logging option for specific debugging')
    args = parser.parse_args()

    network = args.network
    batch_size = args.batch
    arg_devs = list(args.device.split(','))
    use_cpu = use_igp = False

    devices = []
    gpus = []
    cpu = igpu = None
    cpu_idx = igpu_idx = -1

    # Check available devices
    for dev in arg_devs:
        if dev == 'cpu':
            use_cpu = True
            if tvm.cpu(0).exist:
                cpu = Device('cpu', 0)
            else: print('[Error] Device \'%s\' is unrecognizable' % dev)

        elif dev == 'igpu':
            use_igp = True
            if tvm.opencl(0).exist:
                igpu = Device('igpu', 0)
            else: print('[Error] Device \'%s\' is unrecognizable' % dev)

        elif dev.find('gpu') >= 0:
            idx_start = dev.find('gpu') + len('gpu')
            gpu_idx = int(dev[idx_start:])
            if tvm.gpu(gpu_idx).exist:
                gpus.append(Device('gpu', gpu_idx))
            else: print('[Error] Device \'%s\' is unrecognizable' % dev)

    # Add devices
    if cpu is not None:
        cpu_idx = len(devices)
        devices.append(cpu)
    if igpu is not None:
        igpu_idx = len(devices)
        devices.append(igpu)
    if len(gpus) > 0:
        devices.extend(gpus)

    if args.log == 'enable': logging = True
    else: logging = False

    if batch_size == 0 or len(devices) == 0:
        parser.print_help(sys.stderr)
        exit(1)

    env = Environment(network, batch_size, devices, logging)
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

    # Schedule build order for effective sequential builds
    if igpu_idx >= 0:
        env.devices.insert(len(env.devices)-1, env.devices.pop(igpu_idx))
    if cpu_idx >= 0:
        env.devices.insert(len(env.devices)-1, env.devices.pop(cpu_idx))

    result = ''
    for dev in env.devices:
        if dev.batch_size == 0: continue

        # build_time = time.time()
        net, params, input_shape, output_shape = \
            env.get_network(name=env.network, batch_size=dev.batch_size)
        with relay.build_config(opt_level=env.opt_level):
            graph, lib, params = relay.build(net, target=dev.target, params=params)
        # build_time = time.time() - build_time
        # print('<%s> build time: %.3f sec' % (dev.name, build_time))

        t = threading.Thread(target=dev.Run, args=(graph, lib, params, input_shape, env.run_times))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    elapsed_time = time.time() - elapsed_time
    # print('\nPartitioned Result:', work_sizes)
    print(result)
    print('All elapsed time: %.2f sec' % (elapsed_time))
    print('Partitioning time: %.2f ms' % (div_time * 1000))
