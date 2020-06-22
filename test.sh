batch=200
log_name="log_200_10x.xlsx"

python3 Inference.py --network=squeezenet_v1.1 --batch=$batch --device=cpu,igpu,gpu0, --log=$log_name

python3 Inference.py --network=mobilenet --batch=$batch --device=cpu,igpu,gpu0, --log=$log_name

python3 Inference.py --network=vgg-16 --batch=$batch --device=cpu,igpu,gpu0, --log=$log_name

python3 Inference.py --network=vgg-19 --batch=$batch --device=cpu,igpu,gpu0, --log=$log_name

python3 Inference.py --network=resnet-34 --batch=$batch --device=cpu,igpu,gpu0, --log=$log_name

python3 Inference.py --network=resnet-50 --batch=$batch --device=cpu,igpu,gpu0, --log=$log_name

python3 Inference.py --network=inception_v3 --batch=$batch --device=cpu,igpu,gpu0, --log=$log_name
