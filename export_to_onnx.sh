weaver -c /afs/cern.ch/work/m/mgarciam/private/dataconfig/fccee_flavtagging_edm4hep_test_new.add39116ddabf02861ccf8df9aa30cbc.auto.yaml \
 -n /afs/cern.ch/work/m/mgarciam/private/particle_transformer/particle_transformer/networks/example_ParticleNet.py \
 -m /tmp/models_dolores/PNET_2048_09_01_2023/_best_epoch_state.pt \
 --export-onnx /afs/cern.ch/work/m/mgarciam/public/onnx_exports/fccee_flavtagging_edm4hep_wc_v1.onnx \

 cp /afs/cern.ch/work/m/mgarciam/private/dataconfig/fccee_flavtagging_edm4hep_test_new.add39116ddabf02861ccf8df9aa30cbc.auto.yaml  \
 /afs/cern.ch/work/m/mgarciam/public/onnx_exports/fccee_flavtagging_edm4hep_wc_v1.yaml

 mv /afs/cern.ch/work/m/mgarciam/public/onnx_exports/preprocess.json /afs/cern.ch/work/m/mgarciam/public/onnx_exports/fccee_flavtagging_edm4hep_wc_v1.json


