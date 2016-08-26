#!/bin/bash
. ./cmd.sh ## You'll want to change cmd.sh to something that will work on your system.
           ## This relates to the queue.

expindex=dnn_rbm_7l_fbank_40      # experiment index

#Now begin train DNN systems on multi data
. ./path.sh
echo "using RBM pretrain"
#RBM pretrain
exp=exp-mono-dnn
feats=data-fbank
dir=$exp/${expindex}_pretrain
$cuda_cmd $dir/_pretrain_dbn.log \
  steps/nnet/pretrain_dbn.sh \
  --nn-depth 7 --rbm-iter 5 --hid_dim 1024 \
  $feats/train $dir
dbn=$exp/${expindex}_pretrain/7.dbn

dir=$exp/${expindex}
ali=exp/mono_ali
feature_transform=$exp/${expindex}_pretrain/final.feature_transform
$cuda_cmd $dir/_train_nnet.log \
    steps/nnet/train.sh --feature-transform $feature_transform --dbn $dbn --hid-layers 0 --learn-rate 0.008 \
    $feats/train_tr90 $feats/train_cv10 data/lang $ali $ali $dir || exit 1;

#Make graph and decode
echo "make graph and decode for average"
#utils/mkgraph.sh data/lang $dir $dir/graph || exit 1;
steps/nnet/decode.sh --nj 8 --acwt 0.10 --config conf/decode_dnn.config \
  exp/mono/graph $feats/test $dir/decode || exit 1;

