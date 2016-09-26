#!/bin/bash
. ./cmd.sh ## You'll want to change cmd.sh to something that will work on your system.
           ## This relates to the queue.


#Now begin train CNN systems on multi data
. ./path.sh

hid_layers=2
dir=exp/cnn1
ali=exp/tri2b_multi_ali_si84
ali_dev=exp/tri2b_multi_ali_dev_0330
feats=data-fbank
# Train
$cuda_cmd $dir/_train_nnet.log \
  steps/nnet/train.sh \
    --cmvn-opts "--norm-means=false --norm-vars=false" \
    --delta-opts "--delta-order=0" --splice 5 \
    --network-type cnn1d --cnn-proto-opts "--patch-dim1 8 --pitch-dim 3" \
    --hid-layers $hid_layers --learn-rate 0.008 \
      $feats/train_si84_multi $feats/dev_0330 data/lang $ali $ali_dev $dir || exit 1;

#make graph and decode for ABCD
echo "make graph and decode for ABCD"
utils/mkgraph.sh data/lang_test_tgpr_5k $dir $dir/graph_tgpr_5k || exit 1;
for x in test_A test_B test_C test_D;do
  steps/nnet/decode.sh --nj 8 --acwt 0.10 --config conf/decode_dnn.config \
    ${dir}/graph_tgpr_5k data-fbank/$x $dir/decode_tgpr_5k_$x || exit 1;
done

