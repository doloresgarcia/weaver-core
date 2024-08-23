#!/bin/bash
NUM_GPU=${1:-1}
NUM_EPOCH=${2:-40} # default 40

start_time=$(date +%s)

#singularity exec -B /eos -B /afs --nv /afs/cern.ch/work/s/saaumill/public/colorsinglet.sif 

# overcome RAM limitations when running locally - needed???
#ulimit -S -s unlimited

# start training
dirpath="/afs/cern.ch/work/s/saaumill/public"
datapath="/eos/experiment/fcc/ee/datasets/CLD_fullsim_tagging_input/*.root"

if [ $NUM_GPU -eq 1 ]; then # train on one gpu
    singularity exec -B /eos -B /afs --nv /afs/cern.ch/work/s/saaumill/public/colorsinglet.sif python3 "${dirpath}/weaver-core/weaver/train.py" --data-train $datapath --data-config "${dirpath}/weaver-core/configs/example.yaml" --network-config "${dirpath}/particle_transformer/networks/example_ParticleTransformer.py" --model-prefix ./model_weights/ --num-workers 0 --gpus 0 --batch-size 2048 --start-lr 1e-3 --num-epochs $NUM_EPOCH --optimizer ranger --fetch-step 0.01  --log-wandb --wandb-displayname test_fullsim_CLD --wandb-projectname CLD_FullSim_tagging --lr-scheduler reduceplateau
else # train on 4 gpus (must have at least 4 input files)
    singularity exec -B /eos -B /afs --nv /afs/cern.ch/work/s/saaumill/public/colorsinglet.sif torchrun --standalone --nnodes=1 --nproc_per_node=4 "${dirpath}/weaver-core/weaver/train.py" --data-train $datapath --data-config "${dirpath}/weaver-core/configs/example.yaml" --network-config "${dirpath}/particle_transformer/networks/example_ParticleTransformer.py" --model-prefix ./model_weights/ --num-workers 0 --gpus 0,1,2,3 --batch-size 2048 --start-lr 1e-3 --num-epochs $NUM_EPOCH --optimizer ranger --fetch-step 0.01  --log-wandb --wandb-displayname test_fullsim_CLD --wandb-projectname CLD_FullSim_tagging --lr-scheduler reduceplateau --backend nccl
fi

end_time=$(date +%s)
execution_time=$((end_time - start_time))
# Calculate hours, minutes
hours=$((execution_time / 3600))
minutes=$(((execution_time % 3600) / 60))
echo "Execution time: $hours h: $minutes"
echo "Script ran successfully!"
