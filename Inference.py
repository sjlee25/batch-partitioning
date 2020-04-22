import numpy as np
import tvm
from tvm import relay
from tvm.contrib import graph_runtime
from tvm.relay import testing
import threading
import sys
import time
from math import floor
from Partitioning import Partitioner
from os import _exit

# Prints usage with arguments
def PrintInfo():
    print('** A simple test tool for divding batch sizes with performance of each devices **')
    print('< Usage > python3 [file.py] [batch size] [#cpus] [#igps] [#gpus] [gpu indices]\n')
    
    print('  [1] batch size : batch size to execute')
    print('  [2] # cpus     : number of CPUs to use')
    print('  [3] # igps     : number of Integrated GPUs to use')
    print('  [4] # gpus     : number of GPUs to use')
    print('  [5] gpu_idx    : gpu indices to use')
    print('                   uses all gpus if they were not given')
    # print('  [6] # threads  : number of threads to use in TVM')

# Batch sizes
batch_size = int(sys.argv[1])

# Number of devices to use
num_cpus = int(sys.argv[2])
num_igps = int(sys.argv[3])
num_gpus = int(sys.argv[4])
gpu_idxs = []
gpu_start_idx = num_cpus + num_igps
num_devices = gpu_start_idx + num_gpus

# Number of threads to use
# num_threads = int(sys.argv[6])
# bind_threads = int(sys.argv[7])
# if num_threads > 0:
#     os.environ["TVM_NUM_THREADS"] = str(num_threads)
# os.environ["TVM_BIND_THREADS"] = str(bind_threads)

opt_level = 3
target_cpu = 'llvm'
target_igp = tvm.target.intel_graphics()
target_gpu = 'cuda'

num_class = 1000
image_shape = (3, 224, 224)

num_devs = [num_cpus, num_igps, num_gpus]
work_sizes = []
threads = []
eval_time = 1

def PrintError(e, type, idx):
    print('\n[Error] Executing with %s %d failed' % (type, idx))
    print(e)
    _exit(1)

# Runs inference thread with CPU
def RunCPU(cpu_idx=0, batch_size=batch_size):
    if batch_size == 0:
        print('CPU %d = 0.00 ms (0.00 ms)' % (cpu_idx))
        return

    data_shape = (batch_size, ) + image_shape
    out_shape = (batch_size, num_class)
    data = np.random.uniform(-1, 1, size=data_shape).astype('float32')

    mod, params = relay.testing.vgg.get_workload(
        num_layers=16, batch_size=batch_size, image_shape=image_shape)

    with relay.build_config(opt_level=opt_level):
        graph, lib, param = relay.build_module.build(mod, target_cpu, params=params)
    ctx = tvm.cpu(cpu_idx)
    tag = 'CPU EXECUTION'

    try: cpu = graph_runtime.create(graph, lib, ctx)
    except Exception as e:
        PrintError(e, 'CPU', cpu_idx)
    else:
        cpu.set_input('data', data, **param)
        cpu = cpu.module
        cpu = cpu.time_evaluator('run', ctx, 1, eval_time)

        prof_res = np.array(cpu().results) * 1000

        if num_devices > 1:
            print('CPU %d = %7.2f ms (%7.2f ms)  |  %7.2f ms [%3d]' 
                % (cpu_idx, np.mean(prof_res), np.std(prof_res), partitioner.estimated_time[cpu_idx], batch_size))
        else:
            print('CPU %d = %7.2f ms (%7.2f ms)' % (cpu_idx, np.mean(prof_res), np.std(prof_res)))

# Runs inference thread with intel Integrated GPU
def RunIGP(igp_idx=0, batch_size=batch_size):
    if batch_size == 0:
        print('IGP %d = 0.00 ms (0.00 ms)' % (igp_idx))
        return

    data_shape = (batch_size, ) + image_shape
    out_shape = (batch_size, num_class)
    data = np.random.uniform(-1, 1, size=data_shape).astype('float32')

    mod, params = relay.testing.vgg.get_workload(
        num_layers=16, batch_size=batch_size, image_shape=image_shape)

    with relay.build_config(opt_level=opt_level):
        graph, lib, param = relay.build_module.build(mod, target_igp, params=params)
    ctx = tvm.opencl(igp_idx)
    tag = 'IGP EXECUTION'

    try: igp = graph_runtime.create(graph, lib, ctx)
    except Exception as e:
        PrintError(e, 'IGP', igp_idx)
    else:
        igp.set_input('data', data, **param)
        igp = igp.module
        igp = igp.time_evaluator('run', ctx, 1, eval_time)
        prof_res = np.array(igp().results) * 1000

        if num_devices > 1:
            print('IGP %d = %7.2f ms (%7.2f ms)  |  %7.2f ms [%3d]' 
                % (igp_idx, np.mean(prof_res), np.std(prof_res), partitioner.estimated_time[num_cpus + igp_idx], batch_size))
        else:
            print('IGP %d = %7.2f ms (%7.2f ms)' % (igp_idx, np.mean(prof_res), np.std(prof_res)))

# Runs inference thread with GPU
def RunGPU(gpu_idx=0, batch_size=batch_size):
    if batch_size == 0:
        print('GPU %d = 0.00 ms (0.00 ms)' % (gpu_idx))
        return

    data_shape = (batch_size, ) + image_shape
    out_shape = (batch_size, num_class)
    data = np.random.uniform(-1, 1, size=data_shape).astype('float32')

    mod, params = relay.testing.vgg.get_workload(
        num_layers=16, batch_size=batch_size, image_shape=image_shape)

    with relay.build_config(opt_level=opt_level):
        graph, lib, param = relay.build_module.build(mod, target_gpu, params=params)
    ctx = tvm.gpu(gpu_idx)
    tag = 'GPU EXECUTION'

    try: gpu = graph_runtime.create(graph, lib, ctx)
    except Exception as e:
        PrintError(e, 'GPU', gpu_idx)
    else:
        gpu.set_input('data', data, **param)
        gpu = gpu.module
        gpu = gpu.time_evaluator('run', ctx, 1, eval_time)
        prof_res = np.array(gpu().results) * 1000

        if num_devices > 1:
            print('GPU %d = %7.2f ms (%7.2f ms)  |  %7.2f ms [%3d]' 
                % (gpu_idx, np.mean(prof_res), np.std(prof_res), partitioner.estimated_time[num_cpus + num_igps + gpu_idx], batch_size))
        else:
            print('GPU %d = %7.2f ms (%7.2f ms)' % (gpu_idx, np.mean(prof_res), np.std(prof_res)))

if __name__ == '__main__':
    # Guide usage if no argument was given
    needed_args = 5
    if len(sys.argv) < needed_args:
        PrintInfo()
        exit()

    elapsed_time = time.time()

    # GPU indices to use
    if len(sys.argv) == needed_args:
        for i in range(num_gpus):
            gpu_idxs.append(i)
    
    else:
        for i in range(num_gpus):
            if len(sys.argv) - num_gpus != needed_args:
                PrintInfo()
                exit()
            gpu_idxs.append(int(sys.argv[needed_args + i]))
    
    div_time = time.time()
    if num_devices == 1:
        for n in num_devs:
            if n != 0:
                work_sizes.append(batch_size)
                break
    else:
        partitioner = Partitioner(batch_size, num_devs, gpu_idxs)
        partitioner.StartPartition()
        work_sizes = partitioner.partition
        div_time = time.time() - div_time - partitioner.benchmark_time

    if num_cpus == 1:
        t = threading.Thread(target=RunCPU, args=(i, work_sizes[0]))
        threads.append(t)
    if num_igps == 1:
        t = threading.Thread(target=RunIGP, args=(i, work_sizes[num_cpus]))
        threads.append(t)
    for i in range(num_gpus):
        t = threading.Thread(target=RunGPU, args=(i, work_sizes[gpu_start_idx+i]))
        threads.append(t)

    for t in threads:
        t.start()
    for t in threads:
        t.join()

    elapsed_time = time.time() - elapsed_time
    print('\nPartitioned Result:', work_sizes)
    print('All elapsed time: %.2f sec' % (elapsed_time))
    print('Partitioning time: %.2f sec' % (div_time))
