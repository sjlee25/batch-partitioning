import copy
import pickle
import sys

if __name__ == '__main__':
    pop_cnt = 0

    with open('perf_table', 'rb') as file:
        perf_dict = pickle.load(file)
        new_dict = copy.deepcopy(perf_dict)
        dev_name = sys.argv[1]
        network = sys.argv[2]
        print('Pop %s data of %ss from the table...' % (network, dev_name))

        for dev in perf_dict:
            if dev_name in dev:
                if network == 'all':
                    new_dict.pop(dev, None)
                    pop_cnt += 1
                else:
                    if network in perf_dict[dev]:
                        new_dict[dev].pop(network, None)
                        pop_cnt += 1

    if pop_cnt > 0:
        with open('perf_table', 'wb') as file:
            pickle.dump(new_dict, file)
        print(new_dict.keys())
    
    else:
        print('Failed: No %s (%s) in the table' % (dev_name, network)
)