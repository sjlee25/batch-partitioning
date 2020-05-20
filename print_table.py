import pickle
import sys

file = open(sys.argv[1], 'rb')
table = pickle.load(file)

for dev in table:
    print('Device: %s' % (dev))
    dev_dict = table[dev]
    net_list = list(dev_dict.keys())
    net_list.sort()

    for net in net_list:
        print('  Network: %s' % (net))
        keys = sorted(dev_dict[net].keys())
        for i in keys:
            print('    [%3d] %6.2f ms/batch' % (i, dev_dict[net][i]))
        print()

