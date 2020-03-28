'''
    Performance Variation-Aware Partitioning
    Reference: Transparent CPU-GPU Collaboration for
              Data-Parallel Kernels on Heterogeneous Systems
'''
from math import ceil

class CandidateDevice:
    def __init__(self, trial_cnt, diff):
        self.trial_cnt = trial_cnt
        self.diff = diff

def EstimateDevTime(dev_idx):
    dev_time = 0.0
    return dev_time

def EstimateAllTime():
    devs_time = []
    for i in range(enable_devs):
        devs_time.append(EstimateDevTime(i))
    return devs_time

def FindMaxDiffIdx(candidate_devs):
    max_val = candidate_devs[0].diff
    max_idx = 0

    for i in range(1, len(candidate_devs)):
        new_val = candidate_devs[i].diff
        if new_val > max_val:
            max_val = new_val
            max_idx = i
    
    return max_idx

enable_devs = 3
batch_size = 200
max_steps = 1024
granularity = ceil(batch_size/max_steps)

partition = []
for _ in range(enable_devs):
    partition.append(0)
    candidate_devs.append(CandidateDevice(0, float("inf")))

base_dev = 2 # choose the best gpu
partition[base_dev] = batch_size
prev_exec_time = EstimateDevTime(base_dev)

tolerate_cnt = 0
offloaded_cnt = 1

while offloaded_cnt > 0 or tolerate_cnt < 10:
    candidate_devs = []
    offloaded_cnt = 0

    for i in range(enable_devs):
        candidate_devs[i].trial_cnt = 0
        candidate_devs[i].diff = float("inf")

    for i in range(enable_devs):
        offloading_trial = granularity * pow(2, tolerate_cnt)
        if offloading_trial > partition[base_dev]:
            continue

        partition[base_dev] -= offloading_trial
        partition[i] += offloading_trial
        devs_time = EstimateAllTime()
        cur_exec_time = min(devs_time)

        if cur_exec_time < prev_exec_time:
            candidate_devs[i].trial_cnt = offloading_trial
            candidate_devs[i].diff = devs_time[base_dev] - devs_time[i]
        
        partition[base_dev] += offloading_trial
        partition[i] -= offloading_trial

    offload_dev = FindMaxDiffIdx(candidate_devs)
    offloaded_cnt = candidate_devs[offload_dev].offloading_trial

    partition[offload_dev] += offloaded_cnt
    partition[base_dev] -= offloaded_cnt
    if offloaded_cnt > 0: tolerate_cnt = 0
    else: tolerate_cnt += 1
