import copy
import pickle
import sys

if __name__ == '__main__':
    pop_cnt = 0

    with open('perf_table', 'rb') as file:
        perf_dict = pickle.load(file)
        new_dict = copy.deepcopy(perf_dict)
        dev_type = sys.argv[1].upper()
        print('Pop %ss from the table...' % (dev_type))

        type_str = '[%s] ' % (dev_type)
        for dev in perf_dict:
            if type_str in dev:
                new_dict.pop(dev, None)
                pop_cnt += 1

    if pop_cnt > 0:
        with open('perf_table', 'wb') as file:
            pickle.dump(new_dict, file)
        print(new_dict.keys())
    
    else:
        print('Failed: No %s in the table' % (dev_type))
