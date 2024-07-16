#!/bin/bash

source /afs/cern.ch/user/a/asciandr/.bash_profile
source /afs/cern.ch/user/a/asciandr/.bashrc

### NOT NEEDED
### Specify CUDA devices to run with
### Should be the same as --gpus in the weaver line command
#export CUDA_VISIBLE_DEVICES=1

### (MANUAL for now) activate weaver environment
conda activate weaver

### overcome RAM limitations when running locally
ulimit -S -s unlimited

### NEEDED? IT DOES NOT SEEM SO
#source /cvmfs/sft.cern.ch/lcg/views/LCG_103cuda/x86_64-centos7-gcc11-opt/setup.sh
#source /cvmfs/sft.cern.ch/lcg/views/LCG_103cuda/x86_64-centos9-gcc11-opt/setup.sh

#CPU
#weaver --data-train "/eos/atlas/user/a/asciandr/FCC-ee/mysmaller_out_H*.root"  --data-config "/afs/cern.ch/user/a/asciandr/work/FCC/TAGGER_training/example.yaml" --network-config "/afs/cern.ch/user/a/asciandr/work/FCC/TAGGER_training/example_ParticleTransformer.py" --model-prefix "/afs/cern.ch/user/a/asciandr/work/FCC/TAGGER_training/training/"  --num-workers 4 --gpus "" --batch-size 2048 --start-lr 1e-3 --num-epochs 1 --optimizer ranger --fetch-step 0.01 --log CPU_mysmaller_logs/train.log
#GPU
#baseline
#weaver --data-train "/eos/atlas/user/a/asciandr/FCC-ee/reduced200kjets_inputs/baseline*.root"  --data-config "/afs/cern.ch/user/a/asciandr/work/FCC/TAGGER_training/example.yaml" --network-config "/afs/cern.ch/user/a/asciandr/work/FCC/TAGGER_training/example_ParticleTransformer.py" --model-prefix "/afs/cern.ch/user/a/asciandr/work/FCC/TAGGER_training/baseline_recheck_training/"  --num-workers 0 --gpus 0 --batch-size 64 --start-lr 1e-3 --num-epochs 40 --optimizer ranger --fetch-step 0.01 --log mysmaller_logs/train.log --copy-inputs
#no TOF
#weaver --data-train "/eos/atlas/user/a/asciandr/FCC-ee/reduced200kjets_inputs/baseline*.root"  --data-config "/afs/cern.ch/user/a/asciandr/work/FCC/TAGGER_training/example_noTOF.yaml" --network-config "/afs/cern.ch/user/a/asciandr/work/FCC/TAGGER_training/example_ParticleTransformer.py" --model-prefix "/afs/cern.ch/user/a/asciandr/work/FCC/TAGGER_training/baseline_noTOF_training/"  --num-workers 0 --gpus 0 --batch-size 64 --start-lr 1e-3 --num-epochs 40 --optimizer ranger --fetch-step 0.01 --log mysmaller_logs/train.log --copy-inputs
# TOF=3ps (vs. 30ps standard!)
#weaver --data-train "/eos/atlas/user/a/asciandr/FCC-ee/reduced200kjets_inputs/baseline_TOF3ps*.root"  --data-config "/afs/cern.ch/user/a/asciandr/work/FCC/TAGGER_training/example.yaml" --network-config "/afs/cern.ch/user/a/asciandr/work/FCC/TAGGER_training/example_ParticleTransformer.py" --model-prefix "/afs/cern.ch/user/a/asciandr/work/FCC/TAGGER_training/baseline_TOF3ps_training/"  --num-workers 0 --gpus 0 --batch-size 64 --start-lr 1e-3 --num-epochs 40 --optimizer ranger --fetch-step 0.01 --log mysmaller_logs/train.log --copy-inputs
# Same with 1Mx5 jets (i.3. 5x more stats)!
weaver --data-train "/eos/atlas/user/a/asciandr/FCC-ee/reduced1Mjets_inputs/baseline_out*.root"  --data-config "/afs/cern.ch/user/a/asciandr/work/FCC/TAGGER_training/example_nodNdx.yaml" --network-config "/afs/cern.ch/user/a/asciandr/work/FCC/TAGGER_training/example_ParticleTransformer.py" --model-prefix "/afs/cern.ch/user/a/asciandr/work/FCC/TAGGER_training/baseline_nodNdx_1Mjets_training/"  --num-workers 0 --gpus 0 --batch-size 64 --start-lr 1e-3 --num-epochs 40 --optimizer ranger --fetch-step 0.01 --log mysmaller_logs/train.log --copy-inputs
#MS effects disappear at high momentum
#weaver --data-train "/eos/atlas/user/a/asciandr/FCC-ee/jet_p70_inputs/jet_p70_baseline_H*.root"  --data-config "/afs/cern.ch/user/a/asciandr/work/FCC/TAGGER_training/example.yaml" --network-config "/afs/cern.ch/user/a/asciandr/work/FCC/TAGGER_training/example_ParticleTransformer.py" --model-prefix "/afs/cern.ch/user/a/asciandr/work/FCC/TAGGER_training/jet_p70_baseline_training/"  --num-workers 0 --gpus 0 --batch-size 64 --start-lr 1e-3 --num-epochs 40 --optimizer ranger --fetch-step 0.01 --log mysmaller_logs/train.log --copy-inputs
#weaver --data-train "/eos/atlas/user/a/asciandr/FCC-ee/mysmaller_out_H*.root"  --data-config "/afs/cern.ch/user/a/asciandr/work/FCC/TAGGER_training/example.yaml" --network-config "/afs/cern.ch/user/a/asciandr/work/FCC/TAGGER_training/example_ParticleTransformer.py" --model-prefix "/afs/cern.ch/user/a/asciandr/work/FCC/TAGGER_training/training/"  --num-workers 0 --gpus 0 --batch-size 64 --start-lr 1e-3 --num-epochs 1 --optimizer ranger --fetch-step 0.01 --log mysmaller_logs/train.log

#weaver --data-train "/eos/atlas/user/a/asciandr/FCC-ee/out_H*.root"  --data-config "/afs/cern.ch/user/a/asciandr/work/FCC/TAGGER_training/example.yaml" --network-config "/afs/cern.ch/user/a/asciandr/work/FCC/TAGGER_training/example_ParticleTransformer.py" --model-prefix "/afs/cern.ch/user/a/asciandr/work/FCC/TAGGER_training/training/"  --num-workers 4 --gpus 0 --batch-size 32 --start-lr 1e-3 --num-epochs 2 --optimizer ranger --fetch-step 0.00000001 --log logs/train.log
#weaver --data-train "/eos/atlas/user/a/asciandr/FCC-ee/better_singlehitReso_30*.root"  --data-config "/afs/cern.ch/user/a/asciandr/work/FCC/TAGGER_training/example.yaml" --network-config "/afs/cern.ch/user/a/asciandr/work/FCC/TAGGER_training/example_ParticleTransformer.py" --model-prefix "/afs/cern.ch/user/a/asciandr/work/FCC/TAGGER_training/training/"  --num-workers 4 --gpus 0 --batch-size 32 --start-lr 1e-3 --num-epochs 2 --optimizer ranger --fetch-step 0.00000001 --log logs/train.log
#weaver --data-train "/eos/atlas/user/a/asciandr/FCC-ee/better_singlehitReso_30*.root"  --data-config "/afs/cern.ch/user/a/asciandr/work/FCC/TAGGER_training/example.yaml" --network-config "/afs/cern.ch/user/a/asciandr/work/FCC/TAGGER_training/example_ParticleTransformer.py" --model-prefix "/afs/cern.ch/user/a/asciandr/work/FCC/TAGGER_training/training/"  --num-workers 5 --gpus 1,2,3,4 --batch-size 256 --start-lr 1e-3 --num-epochs 2 --optimizer ranger --fetch-step 0.0001 --log logs/train.log
