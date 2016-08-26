#!/bin/bash

. cmd.sh
. path.sh

exp=exp-mono-dnn/dnn_rbm_7l_fbank_40
dirmdl=exp/mono_ali

lang=
post_dim=133 #cusent 133 48

# SP MA JA GE HI
for lang in SP MA JA GE HI; do
#lat="ark:gunzip -c $exp/decode_$lang/lat.*.gz | "
echo "===============Manipulate Language $lang=============="

echo ">> lattice to posterior"
lattice-to-post --acoustic-scale=0.6 "ark:gunzip -c $exp/decode_$lang/lat.*.gz |" ark:tmp.post

echo ">> posterior to phone level posterior"
post-to-phone-post $dirmdl/final.mdl ark:tmp.post ark:tmp.phonepost

echo ">> convert posterior to feats file"
post-to-feats --post-dim=$post_dim ark:tmp.phonepost ark,t:tmp.feat

echo ">> convert feats to HTK file"
copy-feats-to-htk --output-dir=HTK-features-mono-${lang}-cusent --output-ext=fea  ark:tmp.feat

echo ">> remove tmp files"
rm tmp.post tmp.phonepost tmp.feat
done
