#!/bin/bash

. cmd.sh
. path.sh
set -e # exit on error

# call the next line with the directory where the CSLU data is
# (the argument below is just an example).  This should contain
# subdirectories named as follows:
#    test  train 

echo ">> Prepare data & dictionary & language model"

CSLU=~/CSLU/spanish

local/cusent_data_prep.sh $CSLU
local/cusent_prepare_dict.sh '14'

utils/prepare_lang.sh data/local/dict '26' data/local/lang data/lang

local/cusent_format_data.sh #creat data/train folder

