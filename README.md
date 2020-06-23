## Batch Partitioning

##### Usages of files related to performance-based batch partitioning with TVM

<br/>

#### 1. Inference.py

'Inference.py' is a core executable file for this project.

```bash
python3 Inference.py --network=[network name] --device=[devices] --batch=[batch size]
```

```
network: ['mobilenet', 'squeezenet_v1.0', 'squeezenet_v1.1', 'resnet-18', 'resnet-34', 'resnet-50',
          'inception_v3', 'vgg-16', 'vgg-19', 'densenet-121']
device: ['cpu', 'igpu', 'gpu0', 'gpu1', ...]
batch: any positive integer value within a executable range
```

- **To execute it, some codes in TVM framework must be modified (guidelines will be given later here)**
- You can choose one inference network model in the list above.
- For device argument, write all devices to use successively, '--device=cpu,igpu,gpu0,gpu1' for example.
- Partitioned batch sizes must be executable on each device.
- Logging is not yet implemented.

<br/>

#### 2. print_table.py

'print_table.py' prints the performance table previously saved by 'Inference.py'.

If file name of the table is 'perf_table', then it can be executed as below.

```bash
python3 print_table.py perf_table
```

- Modifying the table is not yet completely implemented.

<br/>

#### 3. measure_io.py

'measure_io' is a temporary test code which measures I/O time of each device.

```bash
python3 measure_io.py --network=[network name] --device=[device] --batch=[batch size] \
                      --max_batch=[maximum batch size] --inc=[increments of batch size]
```

- You can iterate measuring with iteration in an one execution.

- If arguments were given as '--batch=100 --max_batch=200 --inc=10', 

  it iterates 11 times with increasing batch size by 10 at each iteration.
