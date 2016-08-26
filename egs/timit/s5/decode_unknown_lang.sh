#!/bin/bash
. ./cmd.sh ## You'll want to change cmd.sh to something that will work on your system.
           ## This relates to the queue.

expindex=dnn_rbm_7l_fbank_40      # experiment index

#Now begin train DNN systems on multi data
. ./path.sh
exp=exp-mono-dnn
feats=~/kaldi-tensor/egs/CSLU/s5/data-fbank

dir=$exp/${expindex}
ali=exp/mono_ali
lang=

for lang in GE HI JA MA SP;do
steps/nnet/decode.sh --nj 1 --acwt 0.10 --config conf/decode_dnn.config \
  exp/mono/graph $feats/$lang $dir/decode_$lang || exit 1;
done
