import pickle
import sys
from Partition import PerfInfo

for i in range(1, len(sys.argv)):
    file = open(sys.argv[i], 'rb')
    table = pickle.load(file)

    for dev in table:
        print('Device: %s' % (dev))
        dev_dict = table[dev]

        for batch in dev_dict:
            perf_info = dev_dict[batch]
            print('    [%3d] %8.2f ms  (i/o: %7.2f ms)' 
                % (batch, perf_info.exec_time*batch, perf_info.io_time*batch))

        print()
