#!/bin/bash

. cmd.sh
. path.sh
set -e # exit on error

# call the next line with the directory where the CUSENT data is
# (the argument below is just an example).  This should contain
# subdirectories named as follows:
#    test  train 

echo ">> Prepare data & dictionary & language model"

CUSENT=/media/datasets/CUSENT_wav_copy

exp=exp

local/cusent_data_prep.sh $CUSENT
local/cusent_prepare_dict.sh

utils/prepare_lang.sh data/local/dict 'sil' data/local/lang data/lang

local/cusent_format_data.sh

# mfccdir should be some place with a largish disk where you
# want to store MFCC features.   You can make a soft link if you want.
featdir=mfcc

echo ">> Feature extraction -- MFCC "
for x in train test dev; do
  utils/fix_data_dir.sh data/$x
  steps/make_mfcc.sh --nj 4 --cmd "run.pl" data/$x $exp/make_feat/$x $featdir
  steps/compute_cmvn_stats.sh data/$x $exp/make_feat/$x $featdir
done

echo ">> Train mono phone GMM-HMM "
steps/train_mono.sh --nj 4 --cmd "$train_cmd" data/train data/lang $exp/mono

#echo ">> Decode"
utils/mkgraph.sh --mono data/local/lm/fst/lm_trigram.arpa $exp/mono $exp/mono/graph
steps/decode.sh --nj 4 --cmd "$decode_cmd" $exp/mono/graph data/test $exp/mono/decode_test

steps/align_si.sh --nj 4 --cmd "$train_cmd" data/train data/lang $exp/mono $exp/mono_ali




