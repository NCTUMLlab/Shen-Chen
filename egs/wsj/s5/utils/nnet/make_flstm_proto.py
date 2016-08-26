#!/usr/bin/env python

# Copyright 2015  Brno University of Technology (author: Karel Vesely)

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
# THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
# WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
# MERCHANTABLITY OR NON-INFRINGEMENT.
# See the Apache 2 License for the specific language governing permissions and
# limitations under the License.

# Generated Nnet prototype, to be initialized by 'nnet-initialize'.

import sys

###
### Parse options
###
from optparse import OptionParser
#usage="%prog [options] <feat-dim> <num-leaves> >nnet-proto-file"
usage="%prog [options] <feat-dim> <num-leaves> <num-hidden-layers> ( <layer1_dim1> <layer1_dim2>... ) >nnet-proto-file"
parser = OptionParser(usage)
#tcn configure
parser.add_option('--activation-type', dest='activation_type',
                    help='Select type of activation function : (<Sigmoid>|<Tanh>) [default: %default]',
                    default='<Sigmoid>', type='string');
parser.add_option('--hid-bias-mean', dest='hid_bias_mean',
                    help='Set bias for hidden activations [default: %default]',
                    default=-2.0, type='float');
parser.add_option('--hid-bias-range', dest='hid_bias_range',
                    help='Set bias range for hidden activations (+/- 1/2 range around mean) [default: %default]',
                     default=4.0, type='float');
parser.add_option('--param-stddev-factor', dest='param_stddev_factor',
                    help='Factor to rescale Normal distriburtion for initalizing weight matrices [default: %default]',
                    default=0.04, type='float');
#lstm configure
parser.add_option('--num-cells', dest='num_cells', type='int', default=800, 
                   help='Number of LSTM cells [default: %default]');
parser.add_option('--num-recurrent', dest='num_recurrent', type='int', default=512, 
                   help='Number of LSTM recurrent units [default: %default]');
parser.add_option('--num-layers', dest='num_layers', type='int', default=2, 
                   help='Number of LSTM layers [default: %default]');
parser.add_option('--lstm-stddev-factor', dest='lstm_stddev_factor', type='float', default=0.01, 
                   help='Standard deviation of initialization [default: %default]');
#parser.add_option('--param-stddev-factor', dest='param_stddev_factor', type='float', default=0.04, 
#                   help='Standard deviation in output layer [default: %default]');
parser.add_option('--clip-gradient', dest='clip_gradient', type='float', default=5.0, 
                   help='Clipping constant applied to gradients [default: %default]');
#
(o,args) = parser.parse_args()
'''
if len(args) != 2 : 
  parser.print_help()
  sys.exit(1)
'''

index_tcn = args.index('tcn')
index_dnn = args.index('dnn')

assert(index_tcn<=index_dnn or index_tcn!=3)

(feat_dim, num_leaves, num_hid_layers) = map(int,args[0:3])
list_tcn_layer_dim = map(int,args[index_tcn+1:index_dnn])
list_dnn_layer_dim = map(int,args[index_dnn+1:])
num_tcn_layers = len(list_tcn_layer_dim)/2-1
if num_tcn_layers<0:
  num_tcn_layers=0
num_dnn_layers = len(list_dnn_layer_dim)

#if num_dnn_layers==0:
list_dnn_layer_dim.append(num_leaves)
if num_tcn_layers>0 and num_dnn_layers==0:
  assert(list_dnn_layer_dim[-1]==num_leaves)
elif num_tcn_layers>0 and num_dnn_layers!=0:
  assert(list_dnn_layer_dim[0]==list_tcn_layer_dim[-2]*list_tcn_layer_dim[-1])
  assert(list_dnn_layer_dim[-1]==num_leaves)
elif num_tcn_layers<=0:
  assert(list_dnn_layer_dim[0]==feat_dim)
  assert(list_dnn_layer_dim[-1]==num_leaves)
### End parse options

# Check

assert(num_hid_layers >= 1)
assert(feat_dim > 0)
assert(num_leaves > 0)
assert(num_tcn_layers + num_dnn_layers == num_hid_layers)


# Original prototype from Jiayu,
#<NnetProto>
#<Transmit> <InputDim> 40 <OutputDim> 40
#<LstmProjectedStreams> <InputDim> 40 <OutputDim> 512 <CellDim> 800 <ParamScale> 0.01 <NumStream> 4
#<AffineTransform> <InputDim> 512 <OutputDim> 8000 <BiasMean> 0.000000 <BiasRange> 0.000000 <ParamStddev> 0.04
#<Softmax> <InputDim> 8000 <OutputDim> 8000
#</NnetProto>

print "<NnetProto>"
if num_tcn_layers!=0:
  # First TCN component
  print "<TCNComponent> <InputDim> %d <OutputDim> %d <BiasMean> %f <BiasRange> %f <ParamStddev> %f <InputDim1> %d <InputDim2> %d <OutputDim1> %d <OutputDim2> %d" % \
       (feat_dim, list_tcn_layer_dim[2]*list_tcn_layer_dim[3],\
       o.hid_bias_mean, o.hid_bias_range, o.param_stddev_factor, \
       list_tcn_layer_dim[0], list_tcn_layer_dim[1],\
       list_tcn_layer_dim[2],list_tcn_layer_dim[3])
  print "%s <InputDim> %d <OutputDim> %d" % \
        (o.activation_type, list_tcn_layer_dim[2]*list_tcn_layer_dim[3], list_tcn_layer_dim[2]*list_tcn_layer_dim[3])

  # Internal TCN component
  for i in range(1,num_tcn_layers-1):
    print "<TCNComponent> <InputDim> %d <OutputDim> %d <BiasMean> %f <BiasRange> %f <ParamStddev> %f <InputDim1> %d <InputDim2> %d <OutputDim1> %d <OutputDim2> %d" % \
           (list_tcn_layer_dim[2*i]*list_tcn_layer_dim[2*i+1], list_tcn_layer_dim[2*(i+1)]*list_tcn_layer_dim[2*(i+1)+1], \
           o.hid_bias_mean, o.hid_bias_range, o.param_stddev_factor, \
           list_tcn_layer_dim[2*i], list_tcn_layer_dim[2*i+1], list_tcn_layer_dim[2*(i+1)],list_tcn_layer_dim[2*(i+1)+1])
    #print "<InputDim1> %d <InputDim2> %d" % (list_tcn_layer_dim[2*i],list_tcn_layer_dim[2*i+1])
    #print "<OutputDim1> %d <OutputDim2> %d" % (list_tcn_layer_dim[2*(i+1)],list_tcn_layer_dim[2*(i+1)+1])
    print "%s <InputDim> %d <OutputDim> %d" % \
        (o.activation_type, list_tcn_layer_dim[2*(i+1)]*list_tcn_layer_dim[2*(i+1)+1], list_tcn_layer_dim[2*(i+1)]*list_tcn_layer_dim[2*(i+1)+1])
    

  # TCN projection component
  '''
  if num_hid_layers!=1:
    print "<TCNProjectionComponent> <InputDim> %d <OutputDim> %d <BiasMean> %f <BiasRange> %f <ParamStddev> %f <InputDim1> %d <InputDim2> %d" % \
      (list_tcn_layer_dim[-4]*list_tcn_layer_dim[-3], list_tcn_layer_dim[-4]*list_tcn_layer_dim[-3], \
      o.hid_bias_mean, o.hid_bias_range, o.param_stddev_factor,\
      list_tcn_layer_dim[-4], list_tcn_layer_dim[-3])
    projection_output=list_tcn_layer_dim[-4]*list_tcn_layer_dim[-3]
  elif num_hid_layers==1:
    print "<TCNProjectionComponent> <InputDim> %d <OutputDim> %d <BiasMean> %f <BiasRange> %f <ParamStddev> %f <InputDim1> %d <InputDim2> %d" % \
      (list_tcn_layer_dim[-2]*list_tcn_layer_dim[-1], list_tcn_layer_dim[-2]*list_tcn_layer_dim[-1], \
      o.hid_bias_mean, o.hid_bias_range, o.param_stddev_factor,\
      list_tcn_layer_dim[-2], list_tcn_layer_dim[-1])
    projection_output=list_tcn_layer_dim[-2]*list_tcn_layer_dim[-1]
  if num_dnn_layers!=0:
    print "%s <InputDim> %d <OutputDim> %d" % \
      (o.activation_type, list_dnn_layer_dim[0], list_dnn_layer_dim[0])
  '''
projection_output=list_tcn_layer_dim[-2]*list_tcn_layer_dim[-1]
# normally we won't use more than 2 layers of LSTM
if o.num_layers == 1:
    print "<LstmProjectedStreams> <InputDim> %d <OutputDim> %d <CellDim> %s <ParamScale> %f <ClipGradient> %f" % \
        (projection_output, o.num_recurrent, o.num_cells, o.lstm_stddev_factor, o.clip_gradient)
elif o.num_layers == 2:
    print "<LstmProjectedStreams> <InputDim> %d <OutputDim> %d <CellDim> %s <ParamScale> %f <ClipGradient> %f" % \
        (projection_output, o.num_recurrent, o.num_cells, o.lstm_stddev_factor, o.clip_gradient)
    print "<LstmProjectedStreams> <InputDim> %d <OutputDim> %d <CellDim> %s <ParamScale> %f <ClipGradient> %f" % \
        (o.num_recurrent, o.num_recurrent, o.num_cells, o.lstm_stddev_factor, o.clip_gradient)
else:
    sys.stderr.write("make_lstm_proto.py ERROR: more than 2 layers of LSTM, not supported yet.\n")
    sys.exit(1)
print "<AffineTransform> <InputDim> %d <OutputDim> %d <BiasMean> 0.0 <BiasRange> 0.0 <ParamStddev> %f" % \
    (o.num_recurrent, num_leaves, o.param_stddev_factor)
print "<Softmax> <InputDim> %d <OutputDim> %d" % \
    (num_leaves, num_leaves)
print "</NnetProto>"


