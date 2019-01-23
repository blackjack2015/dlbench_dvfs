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


data_dir = args.data_dir
file_list = os.listdir(data_dir)

out_dir = data_dir + '_extract'
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

for i, input_file_name in enumerate(file_list):
    
    file_name = input_file_name.split('.')[0]
    file_parameters = file_name.split('_')
    if file_parameters[-1] != 'power':
        continue

    out_file_name = file_name + '_extract.log'
    out_file_path = os.path.join(out_dir, out_file_name)
    out_file = open(out_file_path, 'w')

    input_file_path = os.path.join(data_dir, input_file_name)
    input_file_f = open(input_file_path, 'r')
    input_file = input_file_f.readlines()

    meas_from_begin = AverageMeter()
    meas_from_end = AverageMeter()
    index_begin = 0
    index_end = len(input_file)

    print("file_name*** :  %s \n" % input_file_name)
    for i in range(3, len(input_file)):
        data_line = input_file[i].split()
        meas_from_begin.update(int(data_line[-1]))

        if ((int(data_line[-1]) - 5000) > meas_from_begin.avg) & (i>50):
            # print("meas_from_begin.avg*** :  %d \n" % meas_from_begin.avg)
            index_begin = i
            break

    for i in range(2, len(input_file) - 2):
        j = len(input_file) - i
        data_line = input_file[j].split()
        meas_from_end.update(int(data_line[-1]))

        if ((int(data_line[-1]) - 5000) > meas_from_end.avg) & (i>50):
            # print("meas_from_end.avg*** :  %d \n" % meas_from_end.avg)
            index_end = j
            break

    for i in range(index_begin, index_end):
        out_file.writelines(input_file[i])

    input_file_f.close()
    out_file.close()
