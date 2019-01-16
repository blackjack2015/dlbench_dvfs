import os,sys
import re
import argparse
import ConfigParser
import json

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='Dry runs before measuring performance')
parser.add_argument('--iterations', type=int, default=200, help='How many benchmark runs to measure performance')
parser.add_argument('-b', '--batch-size', default=64, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
args = parser.parse_args()

cfg_file = "configs/torch_config/cnn.cfg"

cfg = ConfigParser.SafeConfigParser()

cfg.read(cfg_file)
datapath = cfg.get('resnet50', 'host143_data_path')

app_exec_cmd = "python torch_imagenet/main.py -a resnet50 --measure resnet50-b%s -b %s  --gpu %s --iterations %s %s" % \
               (args.batch_size, args.batch_size, args.gpu, args.iterations, datapath)
print app_exec_cmd

os.system(app_exec_cmd)

