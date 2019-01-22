import os, sys
import argparse
import re
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


parser = argparse.ArgumentParser(description='GPU measure')
parser.add_argument('-d', '--data-dir', default='p100', type=str,
                    help='sssss')
args = parser.parse_args()

iteration_re = re.compile(r'Iteration')

data_dir = args.data_dir
file_list = os.listdir(data_dir)

out_dir = data_dir + '_extract'
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

for i, input_file_name in enumerate(file_list):

    file_name = input_file_name.split('.')[0]
    file_parameters = file_name.split('_')
    if not ((file_parameters[-1] == 'perf') & (file_parameters[1] == 'caffe')):
        continue

    out_file_name = file_name + '_extract.log'
    out_file_path = os.path.join(out_dir, out_file_name)
    out_file = open(out_file_path, 'w')

    input_file_path = os.path.join(data_dir, input_file_name)
    input_file_f = open(input_file_path, 'r')
    input_file = input_file_f.readlines()

    print("file_name*** :  %s \n" % input_file_name)
    for i in range(len(input_file)):
        data_line = input_file[i]
        if iteration_re.search(data_line):
            out_file.writelines(input_file[i])

    input_file_f.close()
    out_file.close()
