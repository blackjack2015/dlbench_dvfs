import os,sys
import re
import argparse
import ConfigParser
import json

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--iterations', type=int, default=20,
                    help=' how many iterations in the mode of measurement')
parser.add_argument('-t', '--runtime', type=int, default=None,
                    help=' how many seconds in the mode of measurement')
parser.add_argument('-b', '--batch-size', default=64, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')

args = parser.parse_args()

cfg_file = "configs/torch_config/rnn.cfg"

cfg = ConfigParser.SafeConfigParser()

cfg.read(cfg_file)
data_path = cfg.get('ptb', 'host143_data_path')

app_exec_cmd = "python torch_an4/train2.py  --data %s --measure meas -b %s " \
               "--gpu %s --iterations %s -t %s " % \
               (data_path, args.batch_size, args.gpu, args.iterations, args.runtime)

print app_exec_cmd

os.system(app_exec_cmd)
