#!/bin/bash

# Copyright 2013   (Authors: Bagher BabaAli, Daniel Povey, Arnab Ghoshal)
#           2014   Brno University of Technology (Author: Karel Vesely)
#	    2015   Pei-wen Huang -- Verna
# Apache 2.0.

if [ $# -ne 1 ]; then
   echo "Argument should be the CUSENT directory, see ../run.sh for example."
   exit 1;
fi
tmpdir=`pwd`/data/local/tmpdir
dir=`pwd`/data/local/data
lmdir=`pwd`/data/local/nist_lm
mkdir -p $dir $lmdir $tmpdir
local=`pwd`/local
utils=`pwd`/utils
conf=`pwd`/conf

. ./path.sh # Needed for KALDI_ROOT
export PATH=$PATH:$KALDI_ROOT/tools/irstlm/bin
sph2pipe=$KALDI_ROOT/tools/sph2pipe_v2.5/sph2pipe
if [ ! -x $sph2pipe ]; then
   echo "Could not find (or execute) the sph2pipe program at $sph2pipe";
   exit 1;
fi

[ -f $conf/spk_eval.list ] || error_exit "$PROG: Eval-set speaker list not found.";
[ -f $conf/spk_dev.list ] || error_exit "$PROG: dev-set speaker list not found.";

# First check if the train & test directories exist (these can either be upper-
# or lower-cased
if [ ! -d $*/TRAIN -o ! -d $*/TEST ] && [ ! -d $*/train -o ! -d $*/test ]; then
  echo "cusent_data_prep.sh: Spot check of command line argument failed"
  echo "Command line argument must be absolute pathname to CUSENT directory"
  echo "with name like /home/verna/kaldi-trunk/egs/cusent/CUSENT"
  exit 1;
fi

# Now check what case the directory structure is
uppercased=false
train_dir=train
test_dir=test
if [ -d $*/TRAIN ]; then
  uppercased=true
  train_dir=TRAIN
  test_dir=TEST
fi

#tmpdir=$(mktemp -d);
#trap 'rm -rf "$tmpdir"' EXIT

# Get the list of speakers. The list of speakers in the 8-speaker core test 
# set and the 4-speaker development set must be supplied to the script. All
# speakers in the 'train' directory are used for training.
if $uppercased; then
  tr '[:lower:]' '[:upper:]' < $conf/dev_spk.list >$tmpdir/dev_spk
  tr '[:lower:]' '[:upper:]' < $conf/test_spk.list >$tmpdir/test_spk
  ls -d "$*"/TRAIN/* | sed -e "s:^.*/::" > $tmpdir/train_spk
else
  tr '[:upper:]' '[:lower:]' < $conf/dev_spk.list >$tmpdir/dev_spk
  tr '[:upper:]' '[:lower:]' < $conf/test_spk.list >$tmpdir/test_spk
  ls -d "$*"/train/* | sed -e "s:^.*/::" > $tmpdir/train_spk
fi

cd $dir
for x in train dev test; do
  # First, find the list of audio files .
  # Note: train & test sets are under different directories, but doing find on 
  # both and grepping for the speakers will work correctly.
  find $*/{$train_dir,$test_dir} -iname '*.wav' \
    | grep -f $tmpdir/${x}_spk > ${x}_sph.flist

  sed -e 's:.*/\(.*\)/\(.*\).wav$:\1_\2:i' ${x}_sph.flist \
    > $tmpdir/${x}_sph.uttids
  paste $tmpdir/${x}_sph.uttids ${x}_sph.flist \
    | sort -k1,1 > ${x}_sph.scp

  cat ${x}_sph.scp | awk '{print $1}' > ${x}.uttids

  # Create wav.scp
  awk '{printf("%s '$sph2pipe' -f wav %s |\n", $1, $2);}' < ${x}_sph.scp > ${x}_wav.scp

  # Make the utt2spk and spk2utt files.
  cut -f1 -d'_'  ${x}.uttids | paste -d' ' ${x}.uttids - > ${x}.utt2spk 
  cat ${x}.utt2spk | $utils/utt2spk_to_spk2utt.pl > ${x}.spk2utt || exit 1;

  # Prepare gender mapping
  cat ${x}.spk2utt | awk '{print $1}' | perl -ane 'chomp($_); $g = substr($_, -1, 1); print "$_ $g\n";' > ${x}.spk2gender
  
  # Prepare text
  cp $*/${x}.text .

  # Prepare STM file for sclite:
  wav-to-duration scp:${x}_wav.scp ark,t:${x}_dur.ark || exit 1
  awk -v dur=${x}_dur.ark \
  'BEGIN{ 
     while(getline < dur) { durH[$1]=$2; } 
     print ";; LABEL \"O\" \"Overall\" \"Overall\"";
     print ";; LABEL \"F\" \"Female\" \"Female speakers\"";
     print ";; LABEL \"M\" \"Male\" \"Male speakers\""; 
   } 
   { wav=$1; spk=gensub(/_.*/,"",1,wav); $1=""; ref=$0;
     gender=(substr(spk,5,1) == "f" ? "F" : "M");
     printf("%s 1 %s 0.0 %f <O,%s> %s\n", wav, spk, durH[wav], gender, ref);
   }
  ' ${x}.text >${x}.stm || exit 1

  # Create dummy GLM file for sclite:
  echo ';; empty.glm
  [FAKE]     =>  %HESITATION     / [ ] __ [ ] ;; hesitation token
  ' >${x}.glm
done

echo "Data preparation succeeded"
