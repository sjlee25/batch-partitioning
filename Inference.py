import numpy as np
import tvm
from tvm import relay
from tvm.contrib import graph_runtime
from tvm.relay import testing
import time
import threading
from math import floor

import sys
import os

# Prints usage with arguments
def PrintInfo():
    print('** A simple test tool for divding batch sizes with performance of each devices **')
    print('< Usage > python3 [file.py] [batch size] [test batch] [#cpus] [#igps] [#gpus] [#threads]\n')
    
    print('  [1] batch size : batch size to execute')
    print('  [2] test batch : batch size for performance check')
    print('                   input 0 to execute with the given batch size for all devices')
    print('  [3] # cpus     : number of CPUs to use')
    print('  [4] # igps     : number of Integrated GPUs to use')
    print('  [5] # gpus     : number of GPUs to use')
    print('  [6] # threads  : number of threads to use in TVM')

# Guide usage if no argument was given    
if len(sys.argv) < 7:
    PrintInfo()
    exit()

# Batch sizes
batch_size = int(sys.argv[1])
prof_batch = int(sys.argv[2])

# Number of devices to use
num_cpus = int(sys.argv[3])
num_igps = int(sys.argv[4])
num_gpus = int(sys.argv[5])
num_devices = num_cpus + num_igps + num_gpus

# Number of threads to use
num_threads = int(sys.argv[6])
# bind_threads = int(sys.argv[7])
if num_threads > 0:
    os.environ["TVM_NUM_THREADS"] = str(num_threads)
# os.environ["TVM_BIND_THREADS"] = str(bind_threads)

opt_level = 3
target_cpu = 'llvm'
target_igp = 'opencl' # tvm.target.intel_graphics()
target_gpu = 'cuda'

num_class = 1000
image_shape = (3, 224, 224)

# Returns the global device index
# [ cpu0, igp0, gpu0, gpu1, ... ] in order for example
def DeviceIdx(type, idx):
    if type == 'cpu':
        return idx
    elif type == 'igp':
        return num_cpus + idx
    elif type == 'gpu':
        return num_cpus + num_igps + idx
    else: return -1

# Profile each devices by running with 1 batch
# Returns a relative speed (as 1/time)
def ProfileDevices(idx, mod, params, data):
    if idx < num_cpus:
        with relay.build_config(opt_level=opt_level):
            graph, lib, param = relay.build_module.build(mod, target_cpu, params=params)
        ctx = tvm.cpu(idx)

    elif idx < num_cpus + num_igps:
        with relay.build_config(opt_level=opt_level):
            graph, lib, param = relay.build_module.build(mod, target_igp, params=params)
        ctx = tvm.opencl(idx - num_cpus)

    elif idx < num_devices:
        with relay.build_config(opt_level=opt_level):
            graph, lib, param = relay.build_module.build(mod, target_gpu, params=params)
        ctx = tvm.gpu(idx - num_cpus - num_igps)
    
    else: return -1

    elapsed_time = 0

    graph_mod = graph_runtime.create(graph, lib, ctx)
    graph_mod.set_input('data', data, **param)

    # Calculate input, output time costs in each device

    input_time = time.time()
     #param_nd = np.array(list(param.items()))
    param_nd = []
    '''
    for k,v in param.items():
        param_nd.append(v)
    '''
    #print(param_nd)
    #CKH
    '''
    param_nd = []
    for key,val in param.items():
        #print("Key = ",key)
        #print("Value = ",val)
        param_nd.append(val)
    #print(param_nd)
    '''
    # tvm.nd.array(param_nd, ctx)
    # input_time = time.time() - input_time

    # output_time = time.time()
    # graph_mod.get_output(0)
    # output_time = (time.time() - output_time) * 1000
    # elapsed_time += output_time

    graph_mod = graph_mod.module.time_evaluator('run', ctx, 1, 1)
    elapsed_time += graph_mod().results[0] * 1000
    prof_speeds[idx] = 1/elapsed_time

# Divide batch into smaller sizes
# Faster device gets more batches to sync devices
def DivideWorks():
    if num_devices == 0:
        print('No devices available')
        exit()
    
    elif num_devices == 1:
        work_sizes.append(batch_size)
        return

    if prof_batch == 0:
        for i in range(num_devices):
            work_sizes.append(batch_size)
        return

    speeds_sum = 0
    sizes_sum = 0
    fastest_idx = -1
    fastest_speed = -1
    
    # Set data with trial batch size
    data_shape = (prof_batch, ) + image_shape
    out_shape = (prof_batch, num_class)
    mod, params = relay.testing.vgg.get_workload(
        num_layers=16, batch_size=prof_batch, image_shape=image_shape)
    data = np.random.uniform(-1, 1, size=data_shape).astype('float32')

    # Iterate all devices
    prof_threads = []
    for i in range(num_devices):
        t = threading.Thread(target=ProfileDevices, args=(i, mod, params, data))
        prof_threads.append(t)
        t.start()
    for t in prof_threads:
        t.join()

    for i in range(num_devices):
        current_speed = prof_speeds[i]
        speeds_sum += current_speed
        if current_speed > fastest_speed:
            fastest_idx = i
            fastest_speed = current_speed

    # Calculate batch sizes proportional to speed of each device
    for i in range(num_devices):
        prof_speeds[i] /= speeds_sum
        work_size = floor(batch_size * prof_speeds[i])
        work_sizes.append(work_size)
        sizes_sum += work_size
    
    # print(work_sizes)

    # If some batches are left, give them to the fastest one
    left_batch = batch_size - sizes_sum
    if left_batch > 0:
        # temporary: avoid giving zero batch to CPU
        if work_sizes[0] == 0:
            work_sizes[0] += 1
            left_batch -= 1
        if left_batch > 0:
            work_sizes[fastest_idx] += left_batch

    # print(work_sizes)

# Runs inference thread with CPU
def RunCPU(cpu_idx=0, batch_size=batch_size):
    data_shape = (batch_size, ) + image_shape
    out_shape = (batch_size, num_class)
    data = np.random.uniform(-1, 1, size=data_shape).astype('float32')

    mod, params = relay.testing.vgg.get_workload(
        num_layers=16, batch_size=batch_size, image_shape=image_shape)

    with relay.build_config(opt_level=opt_level):
        graph, lib, param = relay.build_module.build(mod, target_cpu, params=params)
    ctx = tvm.cpu(cpu_idx)
    tag = 'CPU EXECUTION'

    cpu = graph_runtime.create(graph, lib, ctx)
    cpu.set_input('data', data, **param)
    cpu = cpu.module
    cpu = cpu.time_evaluator('run', ctx, 1, 10)

    prof_res = np.array(cpu().results) * 1000
    print('CPU %d = %.2f ms (%.2f ms)' % (cpu_idx, np.mean(prof_res), np.std(prof_res)))

# Runs inference thread with intel Integrated GPU
def RunIGP(igp_idx=0, batch_size=batch_size):
    data_shape = (batch_size, ) + image_shape
    out_shape = (batch_size, num_class)
    data = np.random.uniform(-1, 1, size=data_shape).astype('float32')

    mod, params = relay.testing.vgg.get_workload(
        num_layers=16, batch_size=batch_size, image_shape=image_shape)

    with relay.build_config(opt_level=opt_level):
        graph, lib, param = relay.build_module.build(mod, target_igp, params=params)
    ctx = tvm.opencl(igp_idx)
    tag = 'IGP EXECUTION'
    # print('RunIGP: ', ctx.device_name)

    igp = graph_runtime.create(graph, lib, ctx)
    igp.set_input('data', data, **param)
    igp = igp.module
    igp = igp.time_evaluator('run', ctx, 1, 10)

    prof_res = np.array(igp().results) * 1000
    print('IGP %d = %.2f ms (%.2f ms)' % (igp_idx, np.mean(prof_res), np.std(prof_res)))

# Runs inference thread with GPU
def RunGPU(gpu_idx=0, batch_size=batch_size):
    data_shape = (batch_size, ) + image_shape
    out_shape = (batch_size, num_class)
    data = np.random.uniform(-1, 1, size=data_shape).astype('float32')

    mod, params = relay.testing.vgg.get_workload(
        num_layers=16, batch_size=batch_size, image_shape=image_shape)

    with relay.build_config(opt_level=opt_level):
        graph, lib, param = relay.build_module.build(mod, target_gpu, params=params)
    ctx = tvm.gpu(gpu_idx)
    tag = 'GPU EXECUTION'

    gpu = graph_runtime.create(graph, lib, ctx)
    gpu.set_input('data', data, **param)
    gpu = gpu.module
    gpu = gpu.time_evaluator('run', ctx, 1, 10)

    prof_res = np.array(gpu().results) * 1000
    print('GPU %d = %.2f ms (%.2f ms)' % (gpu_idx, np.mean(prof_res), np.std(prof_res)))

prof_speeds = {}
work_sizes = []
threads = []

div_time = time.time()
DivideWorks()
div_time = time.time() - div_time

print("%.2f sec" % (div_time))
print(work_sizes)

for i in range(num_cpus):
    t = threading.Thread(target=RunCPU, args=(i, work_sizes[DeviceIdx('cpu', i)]))
    threads.append(t)
for i in range(num_igps):
    t = threading.Thread(target=RunIGP, args=(i, work_sizes[DeviceIdx('igp', i)]))
    threads.append(t)
for i in range(num_gpus):
    t = threading.Thread(target=RunGPU, args=(i, work_sizes[DeviceIdx('gpu', i)]))
    threads.append(t)

for t in threads:
    t.start()
for t in threads:
    t.join()