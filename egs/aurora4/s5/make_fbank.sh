#!/bin/bash

. ./cmd.sh ## You'll want to change cmd.sh to something that will work on your system.
           ## This relates to the queue.

# make fbank features
fbankdir=fbank
data_fbank=data-fbank
mkdir -p $data_fbank
echo "make filter bank features for average"
for x in train_si84_clean train_si84_multi dev_0330 dev_1206 test_eval92 test_0166; do
  cp -r data/$x $data_fbank/$x
  steps/make_fbank.sh --nj 10 \
    $data_fbank/$x exp/make_fbank/$x $fbankdir || exit 1;
  steps/compute_cmvn_stats.sh $data_fbank/$x exp/make_fbank/$x $fbankdir || exit 1;
done

# make fbank for ABCD
echo "make filter bank features for ABCD"
for x in test_A test_B test_C test_D; do
  cp -r data/$x data-fbank/$x || exit 1;
  steps/make_fbank.sh --nj 10 \
    $data_fbank/$x exp/make_fbank/$x $fbankdir || exit 1;
  steps/compute_cmvn_stats.sh $data_fbank/$x exp/make_fbank/$x $fbankdir || exit 1;
done


