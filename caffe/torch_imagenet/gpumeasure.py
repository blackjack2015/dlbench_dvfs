import os
import pandas as pd
import argparse
import operator
import random
import re
import argparse

# ==========   python gpumeasure.py -t xxxx -o  xxxxx

parser = argparse.ArgumentParser(description='GPU measure')
parser.add_argument('-t', '--txt', default='measurepipeline', type=str,
                    help='sssss')
parser.add_argument('-o', '--outname', default='gpumeasure', type=str,
                    help='gpumeasure file name')

def gpu_summarize(args):
    txtdir = args.txt
    
    gpu_speed_pattern = re.compile(r'(?<=gpu_speed\s:\[)\d+\.?\d*')
    io_time_pattern = re.compile(r'(?<=io_time\s:\s\[)\d+\.?\d*')
    
    file_list = os.listdir(txtdir)
    length = len(file_list)
    
    outfile = args.outname + '.txt'
    out = open(outfile, 'a')
    for i, file in enumerate(file_list):
        lines = open(os.path.join(txtdir, file), 'r')
        gpu_speed_sum = 0
        io_time_sum = 0
        j = 0
        for _, line in enumerate(lines):
            # print(line)
            gpu_speed_str = gpu_speed_pattern.search(line)
            io_time_str = io_time_pattern.search(line)
            # print(speedstr)
            if gpu_speed_str:
                j += 1
                print(j, io_time_str.group())
                if j > 50 and j < 151:
                    gpu_speed_sum += float(gpu_speed_str.group())
                    io_time_sum += float(io_time_str.group())
                if j == 151:
                    break
        gpu_speed_ave = gpu_speed_sum / 100
        io_time_ave = io_time_sum / 100
        out.writelines(file + ' gpu_speed=' + str(gpu_speed_ave) + ',' + ' io_time=' + str(io_time_ave) + '\n')
        lines.close()
    out.close()

if __name__ == '__main__':
    args = parser.parse_args()
    gpu_summarize(args)
