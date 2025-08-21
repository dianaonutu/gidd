#!/bin/bash

# -----------------------------------
# Need to adjust on which HPC to run the sweep.
# -----------------------------------

job_script="pretrain_snls.job"

nodes_list=(1 2 4 8 16)
gpus_list=(4)

# Submit special case node=1, gpu=1
echo "Submitting special case: nodes=1, gpus=1"
sbatch --nodes=1 --gpus-per-node=1 --export=TOTAL_GPUS=1 "$job_script"

for nodes in "${nodes_list[@]}"; do
  for gpus in "${gpus_list[@]}"; do
    total_gpus=$((nodes * gpus))
    echo "Submitting: nodes=${nodes}, gpus-per-node=${gpus}, total=${total_gpus} gpus"
    
    sbatch \
      --nodes=${nodes} \
      --gpus-per-node=${gpus} \
      --export=TOTAL_GPUS=${total_gpus} \
      "$job_script"
  done
done