import pickle

file = open('perf_table', 'rb')
dict = pickle.load(file)

for k in sorted(dict.keys()):
    print(k)
    keys = sorted(dict[k].keys())
    for i in keys:
        print('%3d: %6.2f ms/batch' % (i, dict[k][i]))
    print()