import sys, os
import shutil
import getpass
import glob,fileinput

################                                                                                                                                                                                 
# Current folder                                                                                                                                                                                 
################                                                                                                                                                                                 

# remove auto yaml to ensure proper re-evalutation of data metrics
#os.system("rm example.36ba76501bbe4d08bb79b44b6e45afbf.auto.yaml")

# everything here is done under the current folder!                                                                                                                                              
current=os.getcwd()
output="./"
outfolder_eos="/afs/cern.ch/user/a/asciandr/work/FCC/TAGGER_training/baseline_nodNdx_1Mjets"
outfolder_local=current+"/output_baseline_nodNdx_1Mjets"
if not os.path.isdir(outfolder_eos): os.mkdir(outfolder_eos)
if not os.path.isdir(outfolder_local): os.mkdir(outfolder_local)
print("{0}".format(outfolder_eos))
print("{0}".format(outfolder_local))
print("")

# base job-option                                                                                                                                                                                
basecondor=current+"/submit_generate.sh"

process_local = outfolder_local + "/subdir_baseline_nodNdx_1Mjets"
process_eos = outfolder_eos + "/subdir_baseline_nodNdx_1Mjets"

if not os.path.isdir(process_local): os.mkdir(process_local)
if not os.path.isdir(process_eos): os.mkdir(process_eos)
os.chdir(process_local)


# copy scripts to local dir                                                                                                                                                                  
os.system("cp {0} .".format(basecondor))
# make the condor submission script                                                                                                                                                          
condorscript="{0}".format("MyCondor")
fcondor=open(condorscript,"w")
fcondor.write("Executable ={0}/{1}\n".format(process_local, basecondor.split("/")[-1]))
fcondor.write("Universe = vanilla\n")
fcondor.write("Notification = never\n")
fcondor.write("\n")
fcondor.write("should_transfer_files = YES\n")
fcondor.write("when_to_transfer_output = ON_EXIT\n")
fcondor.write("Output = {0}/condor.out\n".format(process_local))
fcondor.write("Error  = {0}/condor.err\n".format(process_local))
fcondor.write("Log    = {0}/condor.log\n".format(process_local))
fcondor.write("stream_output  = True\n")
fcondor.write("stream_error   = True\n")
fcondor.write("transfer_input_files =\n")
fcondor.write("+JobFlavour  = \"nextweek\"\n")
fcondor.write("request_gpus = 1 \n")
fcondor.write("Queue\n")
fcondor.close()

# do the submission                                                                                                                                                                        
os.system("chmod +x {0}/*.sh".format(process_local))
os.system("condor_submit {0}/{1}".format(process_local, condorscript))

# back to initial folder                                                                                                                                                                       
os.chdir(current)

##########       