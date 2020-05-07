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
import pickle
import sys
import threading
import time
from Partition import Partitioner
    
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
                # dev_name = 'Intel(R) Core(TM) i7-9700K CPU @3.60GHz'
                dev_name = 'Intel(R) Core(TM) i7-8700K CPU @3.60GHz'
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

    # Need to think about the race condition
    def PushResult(self):
        global result
        string = '%s %d = %7.2f ms' % (self.dev_type, self.idx, self.exec_time)
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
            print('\n[Error] Executing with %s %d failed' % (type, idx))
            print(e)
            _exit(1)

        data = tvm.nd.array((np.random.uniform(size=input_shape)).astype('float32'))
        module.set_input('data', data, **params)
        timer = module.module.time_evaluator('run', self.ctx, number=1, repeat=repeat_time)

        self.exec_time = np.mean(np.array(timer().results)) * 1000
        if mode != 'test':
            self.PushResult()
            # throughput = self.batch_size / self.exec_time * 1000

            if env.logging:
                # thrput_path = self.dev_type + '_thrput.txt'
                latency_path = env.network + '_' + self.dev_type + '_latency.txt'
                # acc_path = self.dev_type + '_acc.txt'
                # with open(thrput_path, 'a') as log_file:
                #     log_file.write('%f\n' % (throughput))
                with open(latency_path, 'a') as log_file:
                    log_file.write('%f\n' % (self.exec_time))
                # print('[log] (%d) throughput: %.3f  latency: %.3f' % (self.batch_size, throughput, self.exec_time))
                # with open(acc_path, 'a') as log_file:
                #     log_file.write('%f\n' % ((self.exec_time - self.predict_time) / self.exec_time * 100))

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
    devices = []
    arg_devs = list(args.device.split(','))
    use_cpu = False

    for dev in arg_devs:
        if dev == 'cpu':
            use_cpu = True
            devices.append(Device('cpu', 0))
        elif dev == 'igpu':
            devices.append(Device('igp', 0))
        elif dev.find('gpu') >= 0:
            use_gpu = True
            idx_start = dev.find('gpu') + len('gpu')
            gpu_idx = int(dev[idx_start:])
            if tvm.gpu(gpu_idx).exist:
                devices.append(Device('gpu', gpu_idx))
            else:
                print('[Error] Device \'%s\' is unrecognizable' % dev)
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

    if use_cpu:
        for idx in range(len(env.devices)):
            if env.devices[idx].dev_type == 'cpu':
                env.devices.insert(len(env.devices)-1, env.devices.pop(idx))
                break

    result = ''
    for dev in env.devices:
        build_time = time.time()
        net, params, input_shape, output_shape = \
            env.get_network(name=env.network, batch_size=dev.batch_size)
        with relay.build_config(opt_level=env.opt_level):
            graph, lib, params = relay.build(net, target=dev.target, params=params)
        build_time = time.time() - build_time
        print('<%s> build time: %.3f sec' % (dev.name, build_time))
        t = threading.Thread(target=dev.Run, args=(graph, lib, params, input_shape))
        threads.append(t)
        t.start()
    for t in threads:
        t.join()

    elapsed_time = time.time() - elapsed_time
    # print('\nPartitioned Result:', work_sizes)
    print(result)
    print('All elapsed time: %.2f sec' % (elapsed_time))
    print('Partitioning time: %.2f ms' % (div_time * 1000))
