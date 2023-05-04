## script to export model to onnx
# The result of the script must contain 3 files:
# 1) the auto yaml
# 2) the onnx export
# 3) the json 
#mkdir /afs/cern.ch/work/m/mgarciam/public/onnx_exports/20_03_2022_PT

weaver -c /afs/cern.ch/work/m/mgarciam/private/dataconfig/fccee_flavtagging_edm4hep_12_04_2023.9d82203745911e1e2a44d1444c80c804.auto.yaml \
 -n /afs/cern.ch/work/m/mgarciam/private/particle_transformer/particle_transformer/networks/example_ParticleTransformer.py \
 -m /afs/cern.ch/work/m/mgarciam/private/models/12_04_23_PT/_best_epoch_state.pt \
 --export-onnx /afs/cern.ch/work/m/mgarciam/public/onnx_exports/12_04_2022_PT/fccee_flavtagging_edm4hep_wc.onnx \

 cp /afs/cern.ch/work/m/mgarciam/private/dataconfig/fccee_flavtagging_edm4hep_12_04_2023.9d82203745911e1e2a44d1444c80c804.auto.yaml  \
 /afs/cern.ch/work/m/mgarciam/public/onnx_exports/12_04_23_PT/fccee_flavtagging_edm4hep_wc.yaml

 mv /afs/cern.ch/work/m/mgarciam/public/onnx_exports/12_04_23_PT/preprocess.json /afs/cern.ch/work/m/mgarciam/public/onnx_exports/12_04_23_PT/fccee_flavtagging_edm4hep_wc.json


