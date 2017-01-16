Factorized neural network based on KALDI.

1. Add Tensor object (only support 2-4 way) in nnet data structure.
  kaldi/src/cumatrix
  kaldi/src/matrix
2. Implementation of factorized neural network (support 2-way and 3-way inputs).
  kaldi/src/nnet/nnet-tcn-3way-projection.h
  kaldi/src/nnet/nnet-tcn-3way.h
  kaldi/src/nnet/nnet-tcn-projection.h
  kaldi/src/nnet/nnet-tcn.h
