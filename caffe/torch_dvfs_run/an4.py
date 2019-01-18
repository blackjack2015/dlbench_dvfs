import os,sys
import re
import argparse
import ConfigParser
import json

parser = argparse.ArgumentParser(description='DeepSpeech training')
parser.add_argument('--rnn-type', default='gru', help='Type of the RNN. rnn|gru|lstm are supported')
parser.add_argument('--train-manifest', metavar='DIR',
                    help='path to train manifest csv', default='data/train_manifest.csv')
parser.add_argument('--val-manifest', metavar='DIR',
                    help='path to validation manifest csv', default='data/val_manifest.csv')
parser.add_argument('-b','--batch-size', default=20, type=int, help='Batch size for training')
parser.add_argument('--num-workers', default=4, type=int, help='Number of workers used in data-loading')
parser.add_argument('--hidden-size', default=800, type=int, help='Hidden size of RNNs')
parser.add_argument('--hidden-layers', default=5, type=int, help='Number of RNN layers')
parser.add_argument('--rnn-type', default='gru', help='Type of the RNN. rnn|gru|lstm are supported')
parser.add_argument('--epochs', default=70, type=int, help='Number of training epochs')
parser.add_argument('--cuda', dest='cuda', action='store_true', help='Use cuda to train model')
parser.add_argument('--lr', '--learning-rate', default=3e-4, type=float, help='initial learning rate')
parser.add_argument('--learning-anneal', default=1.1, type=float, help='Annealing applied to learning rate every epoch')
parser.add_argument('--checkpoint', dest='checkpoint', action='store_true', help='Enables checkpoint saving of model')
parser.add_argument('--augment', dest='augment', action='store_true', help='Use random tempo and gain perturbations.')

args = parser.parse_args()

cfg_file = "configs/torch_config/rnn.cfg"

cfg = ConfigParser.SafeConfigParser()

cfg.read(cfg_file)
datapath = cfg.get('an4', 'host143_data_path')

app_exec_cmd = "python torch_an4/train.py  --rnn-type %s --hidden-size %s hidden-layer %s --train-manifest %s   --val-manifest %s  --epochs %s --num-workers %s -b %s --learning-anneal %s --cuda --augment --checkpoint" % \
               (args.rnn_type, args.hidden_size, args.hidden_layer, args.train_manifest, args.val_manifest, args.epochs, args.num_workers, args.baych_size,args.learning_anneal,
                args.cuda, args.cuda, args.augment, args.checkpoint)
print app_exec_cmd

os.system(app_exec_cmd)
