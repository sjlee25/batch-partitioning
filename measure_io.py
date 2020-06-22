import tvm
from tvm import relay
from tvm.contrib import graph_runtime
from util import get_network
import argparse
import numpy as np
import sys
import time
# from Inference import Environment, Device
from Inference import Environment

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--network', type=str, choices=
                        ['mobilenet', 'squeezenet_v1.0', 'squeezenet_v1.1', 'resnet-18', 'resnet-34',
                        'resnet-50', 'inception_v3', 'vgg-16', 'vgg-19', 'densenet-121'], 
                        help='Name of neural network model to use')
    parser.add_argument('--device', type=str, default='gpu0',
                        help='Device to use, write just one argument as \'cpu\', \'igpu\' or \'gpu0\'')
    parser.add_argument('--batch', type=int, help='Batch size')
    parser.add_argument('--max_batch', type=int, default=0, help='Max batch size for iterating check')
    parser.add_argument('--inc', type=int, default=0, help='Increment of batch size for iterating check')
    # parser.add_argument('--log', type=str, default='', help='File path for logging')
    args = parser.parse_args()

    batch_size = args.batch
    max_batch = args.max_batch
    increment = args.inc

    if batch_size == 0 or increment < 0:
        parser.print_help(sys.stderr)
        exit(1)
    if max_batch < batch_size:
        max_batch = batch_size

    network = args.network
    dev = args.device

    # Check if device is available
    if dev == 'cpu':
        if tvm.cpu(0).exist:
            # dev = Device('cpu', 0)
            ctx = tvm.cpu(0)
            target = 'llvm -mcpu=core-avx2'
        else:
            print('[Error] Device \'%s\' is unrecognizable' % dev)
            exit(1)

    elif dev == 'igpu':
        if tvm.opencl(0).exist:
            # dev = Device('igpu', 0)
            ctx = tvm.opencl(0)
            target = tvm.target.intel_graphics()
        else:
            print('[Error] Device \'%s\' is unrecognizable' % dev)
            exit(1)

    elif dev.find('gpu') >= 0:
        idx_start = dev.find('gpu') + len('gpu')
        gpu_idx = int(dev[idx_start:])
        if tvm.gpu(gpu_idx).exist:
            # dev = Device('gpu', gpu_idx)
            ctx = tvm.gpu(gpu_idx)
            target = 'cuda -device=1050ti'
        else:
            print('[Error] Device \'%s\' is unrecognizable' % dev)
            exit(1)

    else:
        print('[Error] Device \'%s\' is unrecognizable' % dev)
        exit(1)

    while batch_size <= max_batch:
        # env = Environment(network, batch_size, [dev], '')
        env = Environment(network, batch_size, [], '')
        
        # build graph
        net, params, input_shape, output_shape = \
            get_network(name=env.network, batch_size=env.batch_size)
        with relay.build_config(opt_level=env.opt_level):
            # graph, lib, params = relay.build(net, target=dev.target, params=params)
            graph, lib, params = relay.build(net, target=target, params=params)
        # module = graph_runtime.create(graph, lib, dev.ctx)
        module = graph_runtime.create(graph, lib, ctx)
        
        # input time
        input_time = time.time()
        param_list = []
        for key, val in params.items():
            param_list.append(val.asnumpy())
        for p in param_list:
            # tvm.nd.array(p, ctx=dev.ctx)
            tvm.nd.array(p, ctx=ctx)
            # print(p.shape)
        input_time = (time.time() - input_time) * 1000 # ms

        # output time
        output_time = time.time()
        module.get_output(0)
        output_time = (time.time() - output_time) * 1000 # ms

        print('input : %8.3f ms\noutput: %8.3f ms\n' % (input_time, output_time))

        if increment > 0: batch_size += increment
        else: break
