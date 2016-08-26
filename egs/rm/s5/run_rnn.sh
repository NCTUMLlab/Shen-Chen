#!/bin/bash

# Copyright 2015  Brno University of Technology (Author: Karel Vesely)
# Apache 2.0

# This example script trains a LSTM network on FBANK features.
# The LSTM code comes from Yiayu DU, and Wei Li, thanks!

# Note: With DNNs in RM, the optimal LMWT is 2-6. Don't be tempted to try acwt's like 0.2, 
# the value 0.1 is better both for decoding and sMBR.

. ./cmd.sh
. ./path.sh

# Train the DNN optimizing per-frame cross-entropy.
dir=exp/tri7a_rnn
ali=exp/tri3b_ali
gmm=exp/tri3b
train=data-fbank/train
dev=data-fbank/test


# Train
$cuda_cmd $dir/log/train_nnet.log \
  steps/nnet/train.sh --network-type rnn --learn-rate 0.0001 \
    --cmvn-opts "--norm-means=true --norm-vars=true" --feat-type plain --splice 0 \
    --scheduler-opts "--momentum 0 --halving-factor 0.5" \
    --train-tool "nnet-train-lstm-streams" \
    --train-tool-opts "--num-stream=2 --targets-delay=5" \
    --proto-opts "--num-hidden 256 --num-layers 1 --clip-gradient 25.0" \
  ${train}_tr90 ${train}_cv10 data/lang $ali $ali $dir || exit 1;

# decode (reuse HCLG graph)
steps/nnet/decode.sh --nj 8 --acwt 0.10 --config conf/decode_dnn.config \
  $gmm/graph $dev $dir/decode || exit 1;

echo Success
exit 0

# Getting results [see RESULTS file]
# for x in exp/*/decode*; do [ -d $x ] && grep WER $x/wer_* | utils/best_wer.sh; done
