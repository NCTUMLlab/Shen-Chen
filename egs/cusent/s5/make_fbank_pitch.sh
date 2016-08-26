#!/bin/bash

. ./cmd.sh ## You'll want to change cmd.sh to something that will work on your system.
           ## This relates to the queue.

# This is a shell script, but it's recommended that you run the commands one by
# one by copying and pasting into the shell.

# make fbank features
dir=fbank
data=data-fbank
mkdir -p $data
#echo "make filter bank features for average"
#for x in train_si84_clean train_si84_multi dev_0330 dev_1206 test_eval92 test_0166; do
#  cp -r data/$x $data_fbank/$x
#  steps/make_fbank.sh --nj 10 \
#    $data_fbank/$x exp/make_fbank/$x $fbankdir || exit 1;
#done

# make fbank 
echo "make filter bank features"
for x in train dev test; do
  cp -r data/$x ${data}/$x || exit 1;
  steps/make_fbank_pitch.sh --nj 10 \
    $data/$x exp/make_fbank/$x $dir || exit 1;
  steps/compute_cmvn_stats.sh $data/$x exp/make_fbank/$x $dir || exit 1;
done


