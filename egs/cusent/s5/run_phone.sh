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

echo ">> Train tri-phone GMM-HMM -- tri1 "
#steps tri1 (first triphone pass)
steps/train_deltas.sh --cmd "$train_cmd" 3000 24000 \
  data/train data/lang $exp/mono_ali $exp/tri1

utils/mkgraph.sh data/local/lm/fst/lm_trigram.arpa $exp/tri1 $exp/tri1/graph
steps/decode.sh --nj 4 --cmd "$decode_cmd" \
  $exp/tri1/graph data/test $exp/tri1/decode_test

steps/align_si.sh --nj 4 --cmd "$train_cmd" --use-graphs true \
  data/train data/lang $exp/tri1 $exp/tri1_ali

echo ">> Train tri2a [delta+delta-deltas] "
steps/train_deltas.sh --cmd "$train_cmd" 3000 24000 \
  data/train data/lang $exp/tri1_ali $exp/tri2a

utils/mkgraph.sh data/local/lm/fst/lm_trigram.arpa $exp/tri2a $exp/tri2a/graph
steps/decode.sh --nj 4 --cmd "$decode_cmd" \
  $exp/tri2a/graph data/test $exp/tri2a/decode_test

echo ">> Train tri2b [LDA+MLLT] "
steps/train_lda_mllt.sh --cmd "$train_cmd" \
  --splice-opts "--left-context=3 --right-context=3" \
  3000 24000 data/train data/lang $exp/tri1_ali $exp/tri2b

utils/mkgraph.sh data/local/lm/fst/lm_trigram.arpa $exp/tri2b $exp/tri2b/graph
steps/decode.sh --nj 4 --cmd "$decode_cmd" \
  $exp/tri2b/graph data/test $exp/tri2b/decode_test

steps/align_si.sh --nj 4 --cmd "$train_cmd" --use-graphs true \
  data/train data/lang $exp/tri2b $exp/tri2b_ali

# echo ">> MMI"
# steps/make_denlats.sh --nj 4 --cmd "$train_cmd" \
  # data/train data/lang $exp/tri2b $exp/tri2b_denlats
# steps/train_mmi.sh data/train data/lang $exp/tri2b_ali $exp/tri2b_denlats $exp/tri2b_mmi
# steps/decode.sh --iter 4 --nj 4 --cmd "$decode_cmd" \
   # $exp/tri2b/graph data/test $exp/tri2b_mmi/decode_it4
# steps/decode.sh --iter 3 --nj 4 --cmd "$decode_cmd" \
   # $exp/tri2b/graph data/test $exp/tri2b_mmi/decode_it3

# #Do the same with boosting.
# steps/train_mmi.sh --boost 0.05 data/train data/lang \
   # $exp/tri2b_ali $exp/tri2b_denlats $exp/tri2b_mmi_b0.05
# steps/decode.sh --config conf/decode.config --iter 4 --nj 4 --cmd "$decode_cmd" \
   # $exp/tri2b/graph data/test $exp/tri2b_mmi_b0.05/decode_it4
# steps/decode.sh --config conf/decode.config --iter 3 --nj 4 --cmd "$decode_cmd" \
   # $exp/tri2b/graph data/test $exp/tri2b_mmi_b0.05/decode_it3

# #Do MPE.
# steps/train_mpe.sh data/train data/lang $exp/tri2b_ali $exp/tri2b_denlats $exp/tri2b_mpe
# steps/decode.sh --iter 4 --nj 4 --cmd "$decode_cmd" \
   # $exp/tri2b/graph data/test $exp/tri2b_mpe/decode_it4
# steps/decode.sh --iter 3 --nj 4 --cmd "$decode_cmd" \
   # $exp/tri2b/graph data/test $exp/tri2b_mpe/decode_it3


echo ">> Do LDA+MLLT+SAT and decode "
steps/train_sat.sh 3000 24000 data/train data/lang $exp/tri2b_ali $exp/tri3b
utils/mkgraph.sh data/local/lm/fst/lm_trigram.arpa $exp/tri3b $exp/tri3b/graph
steps/decode_fmllr.sh --nj 4 --cmd "$decode_cmd" \
  $exp/tri3b/graph data/test $exp/tri3b/decode_test
steps/decode_fmllr.sh --nj 4 --cmd "$decode_cmd" \
  $exp/tri3b/graph data/dev $exp/tri3b/decode_dev

steps/align_fmllr.sh --nj 4 --cmd "$train_cmd" --use-graphs true \
  data/train data/lang $exp/tri3b $exp/tri3b_ali

#. ./run_dnn_phone.sh

