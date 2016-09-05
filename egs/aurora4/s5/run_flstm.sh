#!/bin/bash

. ./cmd.sh ## You'll want to change cmd.sh to something that will work on your system.
           ## This relates to the queue.
#Now begin train TCN systems on multi data
. ./path.sh


#train TCN 
dir=exp/tri8c_flstm
ali=exp/tri2b_multi_ali_si84
ali_dev=exp/tri2b_multi_ali_dev_0330
feats=data-fbank
$cuda_cmd $dir/_train_nnet.log \
  steps/nnet/train.sh --network-type flstm --learn-rate 0.0001 \
    --cmvn-opts "--norm-means=false --norm-vars=false" --feat-type plain --splice 5 \
    --scheduler-opts "--momentum 0.9 --halving-factor 0.5" \
    --train-tool "nnet-train-lstm-streams" \
    --train-tool-opts "--num-stream=4 --targets-delay=5" \
    --proto-opts "--num-cells 1024 --num-recurrent 512 --num-layers 2 --clip-gradient 5.0" \
  data-fbank/train_si84_multi data-fbank/dev_0330 data/lang $ali $ali_dev $dir || exit 1;

#make graph and decode for average
utils/mkgraph.sh data/lang_test_tgpr_5k $dir $dir/graph_tgpr_5k || exit 1;
#steps/nnet/decode.sh --nj 8 --acwt 0.10 --config conf/decode_dnn.config \
#  ${dir}/graph_tgpr_5k $feats/test_eval92 $dir/decode_tgpr_5k_eval92 || exit 1;

#make graph and decode for ABCD
for x in test_A test_B test_C test_D;do
steps/nnet/decode.sh --nj 8 --acwt 0.10 --config conf/decode_dnn.config \
  ${dir}/graph_tgpr_5k data-fbank/$x $dir/decode_tgpr_5k_$x || exit 1;
done


