import os, sys
import argparse

parser = argparse.ArgumentParser(description='GPU measure')
parser.add_argument('-d', '--data-dir', default='p100', type=str,
                    help='sssss')
args = parser.parse_args()

data_dir = args.data_dir
file_list = os.listdir(data_dir)


for i, input_file_name in enumerate(file_list):

    newname = input_file_name.replace('.log', '.txt')
    input_file = os.path.join(data_dir, input_file_name)
    out_file = os.path.join(data_dir, newname)
    os.rename(input_file, out_file)
