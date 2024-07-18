# Info

This provides submit files to train ParticleNet with data from `/eos/experiment/fcc/ee/datasets/CLD_fullsim_tagging_input/*.root` on condor gpus. Look at [this documentary](https://batchdocs.web.cern.ch/gpu/index.html)

Here is an exmpale on how to use it: 

- ssh to lxplus or other machine that supports condor
- go into this directory
- do `condor_submit training.sub`
- to monitor the job use `condor_ssh_to_job -auto-retry 9843071` (replace the number with your cluster number)
- check the ouput/error files at `/afs/cern.ch/work/s/saaumill/public/std-condor-training`
