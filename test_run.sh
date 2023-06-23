#! /bin/bash
	
variations = ("ip", "res", "dndx", "tof")
scale_factors = ("0.1", "0.2", "0.5", "1.0", "2.0", "5.0")
for var in ${variations[@]}; do
    for scale in ${scale_factors[@]}; do
    torchrun --standalone --nnodes=1 --nproc_per_node=2 -m weaver.train \
    --data-train /eos/experiment/fcc/ee/jet_flavour_tagging/winter2023/samples_v2/**.root \
    --data-config /afs/cern.ch/work/m/mgarciam/private/dataconfig/${var}_${scale}.yaml \
    --network-config /afs/cern.ch/work/m/mgarciam/private/particle_transformer/particle_transformer/networks/example_ParticleTransformer.py \
    --model-prefix /eos/user/.../models/${var}_${scale}/ \
    --num-workers 0  \
    --gpus 1,2,3,4 \
    --batch-size 2048 \
    --start-lr 1e-3 \
    --num-epochs 60 \
    --optimizer ranger \
    --fetch-step 0.2 \
    --log logs/train.log \
    --log-wandb \
    --wandb-displayname PT_${var}_${scale} \
    --wandb-projectname test_project \
    --lr-scheduler reduceplateau \
    --backend nccl

done

###
