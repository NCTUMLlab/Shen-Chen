#!/bin/bash

. ./cmd.sh ## You'll want to change cmd.sh to something that will work on your system.
           ## This relates to the queue.

# make fbank features
fbankdir=fbank
data_fbank=data-fbank

train=data-fbank/train

mkdir -p $data_fbank
echo "make filter bank features"
for x in test_mar87 test_oct87 test_feb89 test_oct89 test_feb91 test_sep92 train test; do
  cp -r data/$x data-fbank/$x || exit 1;
  steps/make_fbank.sh --nj 10 \
    $data_fbank/$x exp/make_fbank/$x $fbankdir || exit 1;
  steps/compute_cmvn_stats.sh $data_fbank/$x exp/make_fbank/$x $fbankdir || exit 1;
done

utils/subset_data_dir_tr_cv.sh --cv-spk-percent 10 $train ${train}_tr90 ${train}_cv10
