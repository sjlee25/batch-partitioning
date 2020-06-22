batches=$(seq 2 2 200)
log_name="log_cpu_inception.xlsx"

for i in $batches
do
    python3 Inference.py --network=inception_v3 --batch=$i --device=igpu --log=$log_name
done
