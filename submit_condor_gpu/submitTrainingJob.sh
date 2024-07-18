#!/bin/bash
NUM_GPU=${1}
NUM_EPOCH=${2} # default 40

start_time=$(date +%s)

# source env/docker
echo "Activating singularity..."
singularity shell -B /eos -B /afs --nv /afs/cern.ch/work/s/saaumill/public/colorsinglet.sif
echo "Activated!"

# overcome RAM limitations when running locally - needed???
#ulimit -S -s unlimited

# start training
datapath="/eos/experiment/fcc/ee/datasets/CLD_fullsim_tagging_input/*.root"

if [ $NUM_GPU -eq 1 ]; then # train on one gpu
    python3 -m ./../weaver.train --data-train $datapath --data-config ./../configs/example.yaml --network-config ./../../particle_transformer/networks/example_ParticleTransformer.py --model-prefix ./model_weights/ --num-workers 0 --gpus 0 --batch-size 2048 --start-lr 1e-3 --num-epochs $NUM_EPOCH --optimizer ranger --fetch-step 0.01  --log-wandb --wandb-displayname test_fullsim_CLD --wandb-projectname CLD_FullSim_tagging --lr-scheduler reduceplateau
else # train on multiple gpus (must have at least 4 input files)
    torchrun --standalone --nnodes=1 --nproc_per_node=4 -m ./../weaver.train --data-train $datapath --data-config ./../configs/example.yaml --network-config ./../../particle_transformer/networks/example_ParticleTransformer.py --model-prefix ./model_weights/ --num-workers 0 --gpus 0 --batch-size 2048 --start-lr 1e-3 --num-epochs $NUM_EPOCH --optimizer ranger --fetch-step 0.01  --log-wandb --wandb-displayname test_fullsim_CLD --wandb-projectname CLD_FullSim_tagging --lr-scheduler reduceplateau --backend nccl
fi
end_time=$(date +%s)
execution_time=$((end_time - start_time))
# Calculate hours, minutes
hours=$((execution_time / 3600))
minutes=$(((execution_time % 3600) / 60))
echo "Execution time: $hours h: $minutes"
echo "Script ran successfully!"
