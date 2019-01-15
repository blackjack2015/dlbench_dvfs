import os,sys
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='Dry runs before measuring performance')
parser.add_argument('--iterations', type=int, default=200, help='How many benchmark runs to measure performance')
args = parser.parse_args()

app_exec_cmd = "python ../main.py -a resnet50 --measure resnet-b128 --gpu %s -b 128 --iterations %s /home/hpcl/data/imagenet/imagenet_hdf5" % \
               (args.gpu, args.iterations)


os.system(app_exec_cmd)