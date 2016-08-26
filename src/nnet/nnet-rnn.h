// nnet/nnet-rnn.h

// no tucker

#ifndef KALDI_NNET_NNET_RNN_H_
#define KALDI_NNET_NNET_RNN_H_

#include "nnet/nnet-component.h"
#include "nnet/nnet-utils.h"
#include "cudamatrix/cu-math.h"

/*************************************
 * x: input neuron
 * h: hidden neuron the same as output
 * r: recurrent neuron
 * y: output neuron of RNN
 *************************************/

namespace kaldi {
namespace nnet1 {

class RNNComponent : public UpdatableComponent {
 public:
  RNNComponent(int32 input_dim, int32 output_dim) :
    UpdatableComponent(input_dim, output_dim),
    input_dim_(input_dim),
    output_dim_(output_dim),
    nstream_(0),
    clip_gradient_(0.0)
    //, dropout_rate_(0.0)
  { }

  ~RNNComponent()
  { }

  Component* Copy() const { return new RNNComponent(*this); }
  ComponentType GetType() const { return scRNNComponent; }


  void InitData(std::istream &is) {
    // define options
    float param_scale = 0.02;
    // parse config
    std::string token;
    while (!is.eof()) {
      ReadToken(is, false, &token);
	  //if (token == "<CellDim>") ReadBasicType(is, false, &ncell_);
      if (token == "<ClipGradient>") ReadBasicType(is, false, &clip_gradient_);
      else if (token == "<LearnRateCoef>") ReadBasicType(is, false, &learn_rate_coef_);
      else if (token == "<BiasLearnRateCoef>") ReadBasicType(is, false, &bias_learn_rate_coef_);
      else if (token == "<ParamScale>") ReadBasicType(is, false, &param_scale);
      else KALDI_ERR << "Unknown token " << token << ", a typo in config?"
               << " (CellDim|ClipGradient|ParamScale)";
               //<< " (CellDim|ClipGradient|DropoutRate|ParamScale)";
      is >> std::ws;
    }
    
    // init weight and bias (Uniform)

    wr_.Resize(output_dim_, output_dim_, kUndefined);
    wh_.Resize(output_dim_, input_dim_, kUndefined);
    bias_.Resize(output_dim_, kUndefined);

	
	RandUniform(0.0, 2.0 * param_scale, &wr_);
	RandUniform(0.0, 2.0 * param_scale, &wh_);
	RandUniform(0.0, 2.0 * param_scale, &bias_);

    // init delta buffers
    wr_corr_.Resize(output_dim_, output_dim_, kUndefined);
    wh_corr_.Resize(output_dim_, input_dim_, kSetZero);
    bias_corr_.Resize(output_dim_, kSetZero);

    KALDI_ASSERT(clip_gradient_ >= 0.0);
	KALDI_ASSERT(learn_rate_coef_ >= 0.0);
    KALDI_ASSERT(bias_learn_rate_coef_ >= 0.0);
  }

  void ReadData(std::istream &is, bool binary) {
    ExpectToken(is, binary, "<ClipGradient>");
    ReadBasicType(is, binary, &clip_gradient_);
    
    wh_.Read(is, binary);
    wr_.Read(is, binary);
    bias_.Read(is, binary);

    // init delta buffers
    wh_corr_.Resize(output_dim_, input_dim_, kSetZero);
    wr_corr_.Resize(output_dim_, output_dim_, kSetZero);
    bias_corr_.Resize(output_dim_, kSetZero);
  }

  void WriteData(std::ostream &os, bool binary) const {
    WriteToken(os, binary, "<ClipGradient>");
    WriteBasicType(os, binary, clip_gradient_);
    
    wh_.Write(os, binary);
    wr_.Write(os, binary);
    bias_.Write(os, binary);
  }

  int32 NumParams() const {
    return (wh_.NumRows() * wh_.NumCols() +
         wr_.NumRows() * wr_.NumCols() +
         bias_.Dim());
  }

  void GetParams(VectorBase<BaseFloat>* wei_copy) const {
    //wei_copy->Resize(NumParams());
	KALDI_ASSERT(wei_copy->Dim() == NumParams());

    int32 offset, len;
    
    offset = 0; len = wh_.NumRows() * wh_.NumCols();
    wei_copy->Range(offset, len).CopyRowsFromMat(wh_);

    offset += len; len = wr_.NumRows() * wr_.NumCols();
    wei_copy->Range(offset, len).CopyRowsFromMat(wr_);

    offset += len; len = bias_.Dim();
    wei_copy->Range(offset, len).CopyFromVec(bias_);
	
	offset += len;
    KALDI_ASSERT(offset == NumParams());
  }
  
  void GetGradient(VectorBase<BaseFloat>* gradient) const {
    KALDI_ASSERT(gradient->Dim() == NumParams());
    int32 offset, len;

    offset = 0;    len = wh_corr_.NumRows() * wh_corr_.NumCols();
    gradient->Range(offset, len).CopyRowsFromMat(wh_corr_);

    offset += len; len = wr_corr_.NumRows() * wr_corr_.NumCols();
    gradient->Range(offset, len).CopyRowsFromMat(wr_corr_);

    offset += len; len = bias_.Dim();
    gradient->Range(offset, len).CopyFromVec(bias_corr_);
	
	offset += len;
    KALDI_ASSERT(offset == NumParams());
  }
  
  void SetParams(const VectorBase<BaseFloat>& params) {
    KALDI_ASSERT(params.Dim() == NumParams());
    int32 offset, len;

    offset = 0;    len = wh_.NumRows() * wh_.NumCols();
    wh_.CopyRowsFromVec(params.Range(offset, len));

    offset += len; len = wr_.NumRows() * wr_.NumCols();
    wr_.CopyRowsFromVec(params.Range(offset, len));

    offset += len; len = bias_.Dim();
    bias_.CopyFromVec(params.Range(offset, len));

    offset += len;
    KALDI_ASSERT(offset == NumParams());
  }
  
  std::string Info() const {
    return std::string("  ") +
      "\n  wh_  "   + MomentStatistics(wh_) +
      "\n  wr_  "   + MomentStatistics(wr_) +
      "\n  bias_  " + MomentStatistics(bias_);
  }

  std::string InfoGradient() const {
    return std::string("  ") +
      "\n  Gradients:" +
      "\n  wh_corr_  "    + MomentStatistics(wh_corr_) +
      "\n  wr_corr_  "    + MomentStatistics(wr_corr_) +
      "\n  bias_corr_  "  + MomentStatistics(bias_corr_) +
	  "\n  Forward-pass:" +
      "\n  propagate buffer  " + MomentStatistics(propagate_buf_) +
      "\n  Backward-pass:" +
      "\n  Backpropagate buffer  " + MomentStatistics(backpropagate_buf_);
  }

  void TuckerFeedForward(const CuMatrixBase<BaseFloat> &wei1, const CuMatrixBase<BaseFloat> &wei2,
                         const CuMatrixBase<BaseFloat> &in, CuMatrixBase<BaseFloat>* out)
  {
    //KALDI_LOG<<"wei1:"<<"("<<wei1.NumRows()<<","<<wei1.NumCols()<<")";
    //KALDI_LOG<<"wei2:"<<"("<<wei2.NumRows()<<","<<wei2.NumCols()<<")";
    //KALDI_LOG<<"in:("<<in.NumRows()<<","<<in.NumCols()<<")";
    //KALDI_LOG<<"out"<<"("<<out->NumRows()<<","<<out->NumCols()<<")";

    int32 ib = out->NumRows();
    int32 j1 = wei1.NumRows(), i1 = wei1.NumCols();
    int32 j2 = wei2.NumRows(), i2 = wei2.NumCols();
    // give initial size
    CuMatrix<BaseFloat> output(ib, j1 * j2);
    CuMatrix<BaseFloat> temp(ib, j1 * i2);
    // reshape to std::vector
    std::vector< CuSubMatrix<BaseFloat>* > reshaped_input;
    std::vector< CuSubMatrix<BaseFloat>* > reshaped_output;
    std::vector< CuSubMatrix<BaseFloat>* > reshaped_temp;
    // w1_vec and w2_vec used to parallel
    std::vector< CuSubMatrix<BaseFloat>* > w1_vec;
    std::vector< CuSubMatrix<BaseFloat>* > w2_vec;
    CuSubMatrix<BaseFloat>* w1 = new CuSubMatrix<BaseFloat>(wei1,0,j1,0,i1);
    CuSubMatrix<BaseFloat>* w2 = new CuSubMatrix<BaseFloat>(wei2,0,j2,0,i2);
    // initial some vectors
    for(int32 i = 0; i < ib; i++)
    {
      CuSubMatrix<BaseFloat> *temp_reshaped_input = new CuSubMatrix<BaseFloat>(in,i,i1,i2);
      CuSubMatrix<BaseFloat> *temp_reshaped_temp = new CuSubMatrix<BaseFloat>(temp,i,j1,i2);
      CuSubMatrix<BaseFloat> *temp_reshaped_output = new CuSubMatrix<BaseFloat>(output,i,j1,j2);
      w1_vec.push_back(w1);
      w2_vec.push_back(w2);
      reshaped_input.push_back(temp_reshaped_input);
      reshaped_temp.push_back(temp_reshaped_temp);
      reshaped_output.push_back(temp_reshaped_output);
    }
    AddMatMatBatched<BaseFloat>(1.0,reshaped_temp,w1_vec,kNoTrans,reshaped_input,kNoTrans,0.0);
    AddMatMatBatched<BaseFloat>(1.0,reshaped_output,reshaped_temp,kNoTrans,w2_vec,kTrans,0.0);
    out->CopyFromMat(output);
    // delete pointers
    for(std::vector< CuSubMatrix<BaseFloat>* >::iterator i=reshaped_input.begin(); i!=reshaped_input.end(); ++i)
      delete *i;
    for(std::vector< CuSubMatrix<BaseFloat>* >::iterator i=reshaped_output.begin(); i!=reshaped_output.end(); ++i)
      delete *i;
    for(std::vector< CuSubMatrix<BaseFloat>* >::iterator i=reshaped_temp.begin(); i!=reshaped_temp.end(); ++i)
      delete *i;
    delete w1; delete w2;
  }
  
  // note: out_diff should be tensor type 1, this can save some memory and computation
  void TuckerBackProp(const CuMatrixBase<BaseFloat> &wei1, const CuMatrixBase<BaseFloat> &wei2,
                      const CuMatrixBase<BaseFloat> &out_diff, CuMatrixBase<BaseFloat> *in_diff)
  {
    int32 ib = out_diff.NumRows();
    int32 j1 = wei1.NumRows(), i1 = wei1.NumCols();
    int32 j2 = wei2.NumRows(), i2 = wei2.NumCols();
    // given initial size
    CuMatrix<BaseFloat> temp(ib, i1 * j2);
    CuMatrix<BaseFloat> input_diff(ib, i1 * i2);
    // reshape to std::vector
    std::vector< CuSubMatrix<BaseFloat>* > reshaped_temp;
    std::vector< CuSubMatrix<BaseFloat>* > reshaped_input_diff;
    std::vector< CuSubMatrix<BaseFloat>* > reshaped_output_diff;
    // w1_vec and w2_vec used to parallel
    std::vector< CuSubMatrix<BaseFloat>* > w1_vec;
    std::vector< CuSubMatrix<BaseFloat>* > w2_vec;
    CuSubMatrix<BaseFloat>* w1 = new CuSubMatrix<BaseFloat>(wei1,0,j1,0,i1);
    CuSubMatrix<BaseFloat>* w2 = new CuSubMatrix<BaseFloat>(wei2,0,j2,0,i2);
    // initial some vectors
    for(int32 i = 0; i < ib; i++)
    {
      CuSubMatrix<BaseFloat> *temp_reshaped_temp = new CuSubMatrix<BaseFloat>(temp,i,i1,j2);               //i1 * j2
      CuSubMatrix<BaseFloat> *temp_reshaped_output_diff = new CuSubMatrix<BaseFloat>(out_diff,i,j1,j2);    //j1 * j2
      CuSubMatrix<BaseFloat> *temp_reshaped_input_diff = new CuSubMatrix<BaseFloat>(input_diff,i,i1,i2);   //i1 * i2
      w1_vec.push_back(w1);
      w2_vec.push_back(w2);
      reshaped_temp.push_back(temp_reshaped_temp);
      reshaped_output_diff.push_back(temp_reshaped_output_diff);
      reshaped_input_diff.push_back(temp_reshaped_input_diff);
    }
    AddMatMatBatched<BaseFloat>(1.0,reshaped_temp,w1_vec,kTrans,reshaped_output_diff,kNoTrans,0.0);
    AddMatMatBatched<BaseFloat>(1.0,reshaped_input_diff,reshaped_temp,kNoTrans,w2_vec,kNoTrans,0.0);
    
    in_diff->CopyFromMat(input_diff);
    // delete pointers
    for(std::vector< CuSubMatrix<BaseFloat>* >::iterator i=reshaped_temp.begin(); i!=reshaped_temp.end(); ++i)
      delete *i;
    for(std::vector< CuSubMatrix<BaseFloat>* >::iterator i=reshaped_input_diff.begin(); i!=reshaped_input_diff.end(); ++i)
      delete *i;
    for(std::vector< CuSubMatrix<BaseFloat>* >::iterator i=reshaped_output_diff.begin(); i!=reshaped_output_diff.end(); ++i)
      delete *i;
    delete w1; delete w2;
  }

    // note: in and diff should be tensor type 0, it's easy to compute
  void TuckerCalcGrad(const CuMatrixBase<BaseFloat> &wei1, const CuMatrixBase<BaseFloat> &wei2,
                      const CuMatrixBase<BaseFloat> &in, const CuMatrixBase<BaseFloat> &diff,
                      CuMatrixBase<BaseFloat> *gw1, CuMatrixBase<BaseFloat> *gw2,
                      const BaseFloat mmt)
  {
    //KALDI_LOG<<"wei1:("<<wei1.NumRows()<<","<<wei1.NumCols()<<")";
    //KALDI_LOG<<"wei2:("<<wei2.NumRows()<<","<<wei2.NumCols()<<")";
    //KALDI_LOG<<"in:("<<in.NumRows()<<","<<in.NumCols()<<")";
    //KALDI_LOG<<"diff:("<<diff.NumRows()<<","<<diff.NumCols()<<")";
    //KALDI_LOG<<"gw1:("<<gw1->NumRows()<<","<<gw1->NumCols()<<")";
    //KALDI_LOG<<"gw2:("<<gw2->NumRows()<<","<<gw2->NumCols()<<")";
    int32 ib = in.NumRows();
    int32 j1 = wei1.NumRows(), i1 = wei1.NumCols();
    int32 j2 = wei2.NumRows(), i2 = wei2.NumCols();
    // give initial size
    //gw1->Resize(j1, i1);
    //gw2->Resize(j2, i2);
    CuMatrix<BaseFloat> input(ib, j1 * i2);
    CuMatrix<BaseFloat> dif(ib, j1 * j2);
    CuMatrix<BaseFloat> temp_w1(ib, j1 * i2);
    CuMatrix<BaseFloat> temp_w2(ib, j2 * i1);
    CuMatrix<BaseFloat> w1_grad_batch(ib, j1 * i1);
    CuMatrix<BaseFloat> w2_grad_batch(ib, j2 * i2);
    // reshape to std::vector
    std::vector< CuSubMatrix<BaseFloat>* > reshaped_input;
    std::vector< CuSubMatrix<BaseFloat>* > reshaped_diff;
    std::vector< CuSubMatrix<BaseFloat>* > reshaped_temp_w1;
    std::vector< CuSubMatrix<BaseFloat>* > reshaped_temp_w2;
    std::vector< CuSubMatrix<BaseFloat>* > reshaped_w1_grad_batch;
    std::vector< CuSubMatrix<BaseFloat>* > reshaped_w2_grad_batch;
    // w1_vec and w2_vec used to parallel
    std::vector< CuSubMatrix<BaseFloat>* > w1_vec;
    std::vector< CuSubMatrix<BaseFloat>* > w2_vec;
    CuSubMatrix<BaseFloat>* w1 = new CuSubMatrix<BaseFloat>(wei1,0,j1,0,i1);
    CuSubMatrix<BaseFloat>* w2 = new CuSubMatrix<BaseFloat>(wei2,0,j2,0,i2);
    // initial some vectors
    for(int32 i = 0; i < ib; i++)
    {
      CuSubMatrix<BaseFloat> *temp_reshaped_input = new CuSubMatrix<BaseFloat>(in,i,i1,i2);
      CuSubMatrix<BaseFloat> *temp_reshaped_diff = new CuSubMatrix<BaseFloat>(diff,i,j1,j2);
      CuSubMatrix<BaseFloat> *temp_reshaped_temp_w1 = new CuSubMatrix<BaseFloat>(temp_w1,i,j1,i2);                  //j1 * i2
      CuSubMatrix<BaseFloat> *temp_reshaped_temp_w2 = new CuSubMatrix<BaseFloat>(temp_w2,i,j2,i1);                  //j2 * i1
      CuSubMatrix<BaseFloat> *temp_w1_grad_batch = new CuSubMatrix<BaseFloat>(w1_grad_batch,i,j1,i1);               //j1 * i1
      CuSubMatrix<BaseFloat> *temp_w2_grad_batch = new CuSubMatrix<BaseFloat>(w2_grad_batch,i,j2,i2);               //j2 * i2
      w1_vec.push_back(w1);
      w2_vec.push_back(w2);
      reshaped_input.push_back(temp_reshaped_input);
      reshaped_diff.push_back(temp_reshaped_diff);
      reshaped_temp_w1.push_back(temp_reshaped_temp_w1);
      reshaped_temp_w2.push_back(temp_reshaped_temp_w2);
      reshaped_w1_grad_batch.push_back(temp_w1_grad_batch);
      reshaped_w2_grad_batch.push_back(temp_w2_grad_batch);
    }
    // compute gradient
    // we have w1 and w2 gradient
    // compute w1 grad
    AddMatMatBatched<BaseFloat>(1.0,reshaped_temp_w1,reshaped_diff,kNoTrans,w2_vec,kNoTrans,0.0);
    AddMatMatBatched<BaseFloat>(1.0,reshaped_w1_grad_batch,reshaped_temp_w1,kNoTrans,reshaped_input,kTrans,0.0);
    CuVector<BaseFloat> w1_grad_vec_temp(j1 * i1);
    w1_grad_vec_temp.AddRowSumMat(1.0,w1_grad_batch);
    gw1->CopyRowsFromVec(w1_grad_vec_temp);
    gw1->AddMat(mmt,*gw1);
    // compute w2 grad
    AddMatMatBatched<BaseFloat>(1.0,reshaped_temp_w2,reshaped_diff,kTrans,w1_vec,kNoTrans,0.0);
    AddMatMatBatched<BaseFloat>(1.0,reshaped_w2_grad_batch,reshaped_temp_w2,kNoTrans,reshaped_input,kNoTrans,0.0);
    CuVector<BaseFloat> w2_grad_vec_temp(j2 * i2);
    w2_grad_vec_temp.AddRowSumMat(1.0,w2_grad_batch);
    gw2->CopyRowsFromVec(w2_grad_vec_temp);
    gw2->AddMat(mmt,*gw2);
    // delete pointers
    for(std::vector< CuSubMatrix<BaseFloat>* >::iterator i=reshaped_input.begin(); i!=reshaped_input.end(); ++i)
      delete *i;
    for(std::vector< CuSubMatrix<BaseFloat>* >::iterator i=reshaped_diff.begin(); i!=reshaped_diff.end(); ++i)
      delete *i;
    for(std::vector< CuSubMatrix<BaseFloat>* >::iterator i=reshaped_temp_w1.begin(); i!=reshaped_temp_w1.end(); ++i)
      delete *i;
    for(std::vector< CuSubMatrix<BaseFloat>* >::iterator i=reshaped_temp_w2.begin(); i!=reshaped_temp_w2.end(); ++i)
      delete *i;
    for(std::vector< CuSubMatrix<BaseFloat>* >::iterator i=reshaped_w1_grad_batch.begin(); i!=reshaped_w1_grad_batch.end(); ++i)
      delete *i;
    for(std::vector< CuSubMatrix<BaseFloat>* >::iterator i=reshaped_w2_grad_batch.begin(); i!=reshaped_w2_grad_batch.end(); ++i)
      delete *i;
    delete w1; delete w2;
  }

  void ResetLstmStreams(const std::vector<int32> &stream_reset_flag) {
    // allocate prev_nnet_state_ if not done yet,
    if (nstream_ == 0) {
      // Karel: we just got number of streams! (before the 1st batch comes)
      nstream_ = stream_reset_flag.size();
      prev_nnet_state_.Resize(nstream_, 2 * output_dim_, kSetZero);    // 1 for feedforward 1 for back-propagation
      KALDI_LOG << "Running training with " << nstream_ << " streams.";
    }
    // reset flag: 1 - reset stream network state
    KALDI_ASSERT(prev_nnet_state_.NumRows() == stream_reset_flag.size());
    for (int s = 0; s < stream_reset_flag.size(); s++) {
      if (stream_reset_flag[s] == 1) {
        prev_nnet_state_.Row(s).SetZero();
      }
    }
  }
  
  void PropagateFnc(const CuMatrixBase<BaseFloat> &in, CuMatrixBase<BaseFloat> *out) {
    int DEBUG = 0;
    
    // y(t) = sigma (Wr*y(t-1) + Wh*x(t) + b)
    //      = sigma (Wr*Wh*x(t-1) + Wh*Wr*x(t-1) + b)
    //      = sigma (W*x(t-1) + b)
    static bool do_stream_reset = false;
    if (nstream_ == 0) {
      do_stream_reset = true;
      nstream_ = 1; // Karel: we are in nnet-forward, so 1 stream,
      prev_nnet_state_.Resize(nstream_, 2 * output_dim_ , kSetZero);
      KALDI_LOG << "Running nnet-forward with per-utterance RNN-state reset";
    }
    if (do_stream_reset) prev_nnet_state_.SetZero();
    KALDI_ASSERT(nstream_ > 0);
    KALDI_ASSERT(in.NumRows() % nstream_ == 0);
    int32 T = in.NumRows() / nstream_;
    int32 S = nstream_;
    
    // 0:forward pass history, [1, T]:current sequence, T+1:dummy
    propagate_buf_.Resize((T+2)*S, output_dim_, kSetZero);
    propagate_buf_.RowRange(0*S,S).CopyFromMat(prev_nnet_state_.ColRange(0*output_dim_, output_dim_));
    // x(1:T) * wh'
    propagate_buf_.RowRange(1*S,T*S).AddMatMat(1.0, in, kNoTrans, wh_, kTrans, 0.0);
	// add bias
	out->AddVecToRows(1.0, bias_);
    for (int t = 1; t <= T; t++) {
      // x(t) = x(t-1) * wr'
      propagate_buf_.RowRange(t*S,S).AddMatMat(1.0, propagate_buf_.RowRange((t-1)*S,S), kNoTrans, wr_, kTrans, 1.0);
	  propagate_buf_.RowRange(t*S,S).Sigmoid(propagate_buf_.RowRange(t*S,S)); 
      // debug
      if (DEBUG) {
        std::cerr << "forward-pass frame " << t << "\n";
		std::cerr << "propagate_buf_.RowRange(t*S,S) " << propagate_buf_.RowRange(t*S,S);
      }
    }

    // copy propagate buffer to ouput
    out->AddMat(1.0, propagate_buf_.RowRange(1*S,T*S));

    // now the last frame state becomes previous network state for next batch
    prev_nnet_state_.ColRange(0*output_dim_, output_dim_).CopyFromMat(propagate_buf_.RowRange(T*S,S));
  }

  void BackpropagateFnc(const CuMatrixBase<BaseFloat> &in, const CuMatrixBase<BaseFloat> &out,
              const CuMatrixBase<BaseFloat> &out_diff, CuMatrixBase<BaseFloat> *in_diff) {

    int DEBUG = 0;
	const BaseFloat mmt = opts_.momentum;
    int32 T = in.NumRows() / nstream_;
    int32 S = nstream_;

    // 0:dummy, [1,T] frames, T+1 backward pass history
    //diff_buf_.Resize((T+2)*S, output_dim_, kSetZero);             // store the diff from higher layer with no recurrent
	backpropagate_buf_.Resize((T+2)*S, output_dim_, kSetZero);    // store the recurrent diff
	
	CuMatrix<BaseFloat> diff(out.NumRows(),out.NumCols());
	//diff.DiffSigmoid(out, out_diff);
    backpropagate_buf_.RowRange(1*S,T*S).CopyFromMat(diff);
	backpropagate_buf_.RowRange((T+1)*S,S).CopyFromMat(prev_nnet_state_.ColRange(1*output_dim_, output_dim_)); // T+1 backward pass history
	//diff_buf_.RowRange(1*S,T*S).CopyFromMat(diff);
	//diff_buf_.RowRange((T+1)*S,S).CopyFromMat(prev_nnet_state_.ColRange(1*output_dim_, output_dim_)); // T+1 backward pass history
	
    for (int t = T; t >= 1; t--) {
      /// dx(t) = dx(t+1) * wr
      // debug info
	  //diff_buf_.RowRange(t*S,S).DiffSigmoid(propagate_buf_.RowRange((t+1)*S,S), backpropagate_buf_.RowRange((t+1)*S,S));
	  backpropagate_buf_.RowRange(t*S,S).AddMatMat(1.0, backpropagate_buf_.RowRange((t+1)*S,S), kNoTrans, wr_, kNoTrans, 1.0);
	  
      if (DEBUG) {
        //std::cerr << "backward-pass frame " << t << "\n";
		//std::cerr << "backpropagate_buf_.RowRange(t*S,S) " << backpropagate_buf_.RowRange(t*S,S);
      }
    }
	
    in_diff->AddMatMat(1.0, backpropagate_buf_.RowRange(S,T*S), kNoTrans, wh_, kNoTrans, 0.0);
	// now the first error becomes previous network state for next batch
	prev_nnet_state_.ColRange(1*output_dim_, output_dim_).CopyFromMat(backpropagate_buf_.RowRange(1*S,S));
	/////// calculate delta ///////
    /////// gradients       ///////
    //const BaseFloat mmt = opts_.momentum;
    // wh
    wh_corr_.AddMatMat(1.0, diff, kTrans, 
                       in, kNoTrans, mmt);

    // wr
    //wr_corr_.AddMatMat(1.0, backpropagate_buf_.RowRange(1*S,T*S), kTrans, 
    //                   propagate_buf_.RowRange(1*S,T*S), kNoTrans, mmt);
	wr_corr_.AddMatMat(1.0, diff, kTrans, 
                       propagate_buf_.RowRange(1*S,T*S), kNoTrans, mmt);
       
    // bias 
    //bias_corr_.AddRowSumMat(1.0, backpropagate_buf_.RowRange(1*S,T*S), mmt);
	bias_corr_.AddRowSumMat(1.0, diff, mmt);
    
 
    if (clip_gradient_ > 0.0) {
      wh_corr_.ApplyFloor(-clip_gradient_);
      wh_corr_.ApplyCeiling(clip_gradient_);

      wr_corr_.ApplyFloor(-clip_gradient_);
      wr_corr_.ApplyCeiling(clip_gradient_);

      bias_corr_.ApplyFloor(-clip_gradient_);
      bias_corr_.ApplyCeiling(clip_gradient_);
    }

    if (DEBUG) {
      std::cerr << "gradients(with optional momentum): \n";
      std::cerr << "wh_corr_ " << wh_corr_;
      std::cerr << "wr_corr_ " << wr_corr_;
      std::cerr << "bias_corr_ " << bias_corr_;
    }
  }

  void Update(const CuMatrixBase<BaseFloat> &input, const CuMatrixBase<BaseFloat> &diff) {
    const BaseFloat lr  = opts_.learn_rate;

    wh_.AddMat(-lr, wh_corr_);

    wr_.AddMat(-lr, wr_corr_);

    bias_.AddVec(-lr, bias_corr_, 1.0);

  }

 private:
  // dims
  int32 nstream_;

  int32 input_dim_,output_dim_;

  //int32 i1_,i2_;     // input dim 
  //int32 rj1_,rj2_;   // recurrnet layer dim

  CuMatrix<BaseFloat> prev_nnet_state_;

  // gradient-clipping value,
  BaseFloat clip_gradient_;

  // recurrent and feedforward
  CuMatrix<BaseFloat> wr_;
  CuMatrix<BaseFloat> wr_corr_;
  CuMatrix<BaseFloat> wh_;
  CuMatrix<BaseFloat> wh_corr_;

  // biases
  CuVector<BaseFloat> bias_;
  CuVector<BaseFloat> bias_corr_;

  // propagate buffer: output
  CuMatrix<BaseFloat> propagate_buf_;

  // back-propagate buffer: diff-input
  CuMatrix<BaseFloat> backpropagate_buf_;

};
} // namespace nnet1
} // namespace kaldi

#endif
