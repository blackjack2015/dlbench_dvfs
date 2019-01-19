import os,sys
import re
import argparse
import ConfigParser
import json

parser = argparse.ArgumentParser(description='DeepSpeech training')
parser.add_argument('--rnn-type', default='gru', help='Type of the RNN. rnn|gru|lstm are supported')
parser.add_argument('--hidden-size', default=800, type=int, help='Hidden size of RNNs')
parser.add_argument('--hidden-layers', default=5, type=int, help='Number of RNN layers')
parser.add_argument('--cuda', dest='cuda', action='store_true', help='Use cuda to train model')

parser.add_argument('-b', '--batch-size', default=64, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--iterations', type=int, default=20,
                    help=' how many iterations in the mode of measurement')
parser.add_argument('-t', '--runtime', type=int, default=None,
                    help=' how many seconds in the mode of measurement')

args = parser.parse_args()

cfg_file = "configs/torch_config/rnn.cfg"

cfg = ConfigParser.SafeConfigParser()

cfg.read(cfg_file)
train_manifest = cfg.get('an4', 'host143_train_manifest')
val_manifest = cfg.get('an4', 'host143_val_manifest')
rnn = cfg.get('an4', 'rnn_type')

app_exec_cmd = "CUDA_VISIBLE_DEVICES=%s python torch_an4/train.py  --rnn-type %s --hidden-size %s --hidden-layers %s --train-manifest %s  --val-manifest %s " \
               " -b %s --iterations %s -t %s --cuda " % \
               (args.gpu, rnn, args.hidden_size, args.hidden_layers, train_manifest, val_manifest,
                args.batch_size, args.iterations, args.runtime)
# args.cuda, args.cuda, args.augment, args.checkpoint
print app_exec_cmd

os.system(app_exec_cmd)
