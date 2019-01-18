import os,sys
import re
import argparse
import ConfigParser
import json

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=20)
args = parser.parse_args()

cfg_file = "configs/torch_config/rnn.cfg"

cfg = ConfigParser.SafeConfigParser()

cfg.read(cfg_file)
datapath = cfg.get('lstm', 'host143_data_path')

app_exec_cmd = "python torch_ptb/torchtrain.py --batch_size %s  % \
               (args.batch_size)
print app_exec_cmd

os.system(app_exec_cmd)
