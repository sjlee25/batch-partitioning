import pickle

file = open('perf_table', 'rb')
table = pickle.load(file)

for dev in table:
    print('Device: %s' % (dev))
    dev_dict = table[dev]
    for net in dev_dict:
        print('  Network: %s' % (net))

        keys = sorted(dev_dict[net].keys())
        for i in keys:
            print('    [%2d] %6.2f ms/batch' % (i, dev_dict[net][i]))
        print()

