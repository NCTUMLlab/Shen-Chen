#!/bin/bash

. cmd.sh
. path.sh
set -e # exit on error

# call the next line with the directory where the CSLU data is
# (the argument below is just an example).  This should contain
# subdirectories named as follows:
#    test  train 

echo ">> Prepare data & dictionary & language model"

# CSLU=/home/siyuan/kaldi/CSLU/english

# local/cusent_data_prep.sh $CSLU
# local/cusent_prepare_dict.sh '14'

# utils/prepare_lang.sh data/local/dict '26' data/local/lang data/lang

# local/cusent_format_data.sh #creat data/train folder

# mfccdir should be some place with a largish disk where you
# want to store MFCC features.   You can make a soft link if you want.
featdir=mfcc

# echo ">> Feature extraction -- MFCC "
# for x in train ; do
  # # utils/fix_data_dir.sh data/$x
  # steps/make_mfcc.sh --nj 1 --cmd "run.pl" data/$x exp/make_feat/$x $featdir
  # steps/compute_cmvn_stats.sh data/$x exp/make_feat/$x $featdir
# done
it_idx=12;
while [ $it_idx	-lt 15 ]; do
	echo ">> Train mono phone GMM-HMM "
	mkdir -p exp/mono_iter$it_idx
	cp exp/mono_iter$[$it_idx-1]/40.mdl exp/mono_iter$it_idx/1.mdl
	cp exp/mono_iter$[$it_idx-1]/decode_test/scoring_kaldi/penalty_0.0/9.txt data/train/text${it_idx}
	cp exp/mono_iter$[$it_idx-1]/decode_test/scoring_kaldi/penalty_0.0/9.txt data/train/split1/1/text${it_idx}
	steps/train_mono.sh --nj 1 --cmd "$train_cmd" data/train data/lang exp/mono_iter$it_idx $it_idx #temporarily comment 'split_data.sh' line for unsolved problem regarding 'split1/1/cmvn.scp'

	echo ">> Decode"
	utils/mkgraph.sh --mono data/local/lm/fst/count.arpa exp/mono_iter$it_idx exp/mono_iter$it_idx/graph
	steps/decode.sh --nj 1 --cmd "$decode_cmd" exp/mono_iter$it_idx/graph data/train exp/mono_iter$it_idx/decode_test $it_idx
	it_idx=$[$it_idx+1]
done
# steps/align_si.sh --nj 1 --cmd "$train_cmd" data/train data/lang exp/mono exp/mono_ali

# echo ">> Train tri-phone GMM-HMM -- tri1 "
# # steps tri1 (first triphone pass)
# steps/train_deltas.sh --cmd "$train_cmd" 3000 24000 \
  # data/train data/lang exp/mono_ali exp/tri1

# utils/mkgraph.sh data/local/lm/fst/bigram.arpa exp/tri1 exp/tri1/graph
# steps/decode.sh --nj 4 --cmd "$decode_cmd" \
  # exp/tri1/graph data/test exp/tri1/decode_test

# steps/align_si.sh --nj 4 --cmd "$train_cmd" --use-graphs true \
  # data/train data/lang exp/tri1 exp/tri1_ali

# echo ">> Train tri2a [delta+delta-deltas] "
# steps/train_deltas.sh --cmd "$train_cmd" 3000 24000 \
  # data/train data/lang exp/tri1_ali exp/tri2a

# utils/mkgraph.sh data/local/lm/fst/bigram.arpa exp/tri2a exp/tri2a/graph
# steps/decode.sh --nj 4 --cmd "$decode_cmd" \
  # exp/tri2a/graph data/test exp/tri2a/decode_test

# echo ">> Train tri2b [LDA+MLLT] "
# steps/train_lda_mllt.sh --cmd "$train_cmd" \
  # --splice-opts "--left-context=3 --right-context=3" \
  # 3000 24000 data/train data/lang exp/tri1_ali exp/tri2b

# utils/mkgraph.sh data/local/lm/fst/bigram.arpa exp/tri2b exp/tri2b/graph
# steps/decode.sh --nj 4 --cmd "$decode_cmd" \
  # exp/tri2b/graph data/test exp/tri2b/decode_test

# steps/align_si.sh --nj 4 --cmd "$train_cmd" --use-graphs true \
  # data/train data/lang exp/tri2b exp/tri2b_ali

# # echo ">> MMI"
# # steps/make_denlats.sh --nj 4 --cmd "$train_cmd" \
  # # data/train data/lang exp/tri2b exp/tri2b_denlats
# # steps/train_mmi.sh data/train data/lang exp/tri2b_ali exp/tri2b_denlats exp/tri2b_mmi
# # steps/decode.sh --iter 4 --nj 4 --cmd "$decode_cmd" \
   # # exp/tri2b/graph data/test exp/tri2b_mmi/decode_it4
# # steps/decode.sh --iter 3 --nj 4 --cmd "$decode_cmd" \
   # # exp/tri2b/graph data/test exp/tri2b_mmi/decode_it3

# # # Do the same with boosting.
# # steps/train_mmi.sh --boost 0.05 data/train data/lang \
   # # exp/tri2b_ali exp/tri2b_denlats exp/tri2b_mmi_b0.05
# # steps/decode.sh --config conf/decode.config --iter 4 --nj 4 --cmd "$decode_cmd" \
   # # exp/tri2b/graph data/test exp/tri2b_mmi_b0.05/decode_it4
# # steps/decode.sh --config conf/decode.config --iter 3 --nj 4 --cmd "$decode_cmd" \
   # # exp/tri2b/graph data/test exp/tri2b_mmi_b0.05/decode_it3

# # # Do MPE.
# # steps/train_mpe.sh data/train data/lang exp/tri2b_ali exp/tri2b_denlats exp/tri2b_mpe
# # steps/decode.sh --iter 4 --nj 4 --cmd "$decode_cmd" \
   # # exp/tri2b/graph data/test exp/tri2b_mpe/decode_it4
# # steps/decode.sh --iter 3 --nj 4 --cmd "$decode_cmd" \
   # # exp/tri2b/graph data/test exp/tri2b_mpe/decode_it3


# echo ">> Do LDA+MLLT+SAT and decode "
# steps/train_sat.sh 3000 24000 data/train data/lang exp/tri2b_ali exp/tri3b
# utils/mkgraph.sh data/local/lm/fst/bigram.arpa exp/tri3b exp/tri3b/graph
# steps/decode_fmllr.sh --nj 4 --cmd "$decode_cmd" \
  # exp/tri3b/graph data/test exp/tri3b/decode_test
# steps/decode_fmllr.sh --nj 4 --cmd "$decode_cmd" \
  # exp/tri3b/graph data/dev exp/tri3b/decode_dev

# steps/align_fmllr.sh --nj 4 --cmd "$train_cmd" --use-graphs true \
  # data/train data/lang exp/tri3b exp/tri3b_ali



