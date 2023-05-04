#! /bin/bash

source /cvmfs/sft.cern.ch/lcg/contrib/cuda/11.4/x86_64-centos8/setup.sh 
source /afs/cern.ch/work/m/mgarciam/private/miniconda/miniconda3/etc/profile.d/conda.sh
conda activate weaver 
cd /afs/cern.ch/work/m/mgarciam/private/weaver-code-dev-dolo/weaver-core/ 

torchrun --standalone --nnodes=1 --nproc_per_node=2 -m weaver.train  \
--data-train  \
/eos/experiment/fcc/ee/jet_flavour_tagging/winter2023/samples_gen_v1/stage2_Hbb.root \
/eos/experiment/fcc/ee/jet_flavour_tagging/winter2023/samples_gen_v1/stage2_Hcc.root \
/eos/experiment/fcc/ee/jet_flavour_tagging/winter2023/samples_gen_v1/stage2_Hgg.root \
/eos/experiment/fcc/ee/jet_flavour_tagging/winter2023/samples_gen_v1/stage2_Hqq.root \
/eos/experiment/fcc/ee/jet_flavour_tagging/winter2023/samples_gen_v1/stage2_Hss.root \
--data-config /afs/cern.ch/work/m/mgarciam/private/dataconfig/fccee_flavtagging_edm4hep_10_03_23.yaml \
--network-config /afs/cern.ch/work/m/mgarciam/private/particle_transformer/particle_transformer/networks/example_ParticleTransformer.py   \
--model-prefix /afs/cern.ch/work/m/mgarciam/private/models/10_03_23_PT_ideal_MC/   \
--num-workers 4   \
--gpus 0,1  \
--batch-size 2048   \
--start-lr 1e-3   \
--num-epochs 40   \
--optimizer ranger   \
--fetch-step 0.01   \
--log logs/train.log   \
--log-wandb   \
--wandb-displayname ND_PT_2048_idealMC  \
--wandb-projectname test_project  \
--backend nccl  \
#torchrun --standalone --nnodes=1 --nproc_per_node=2 

