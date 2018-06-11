import time
import datetime
import sys,urllib,urllib2
import csv
import os, glob, re
import cPickle as pickle
import numpy as np
import ConfigParser
import json
import pandas as pd

logRoot = 'logs/v100-vgg-logs'
splitword = 'maxwell_scudnn_128x64_relu_small_nn' # for vgg
# splitword = 'maxwell_scudnn_128x64_relu_medium_nn' # for googlenet

keywords = []

metrics_filelist = glob.glob(r'%s/*metrics.log' % logRoot)
metrics_filelist.sort()

coreBase = 0
memBase = 0

head = ["appName", "coreF", "memF", "kernel", "count", "aver_time", "std_time", "coeff. of var."]

# prepare csv file
csvfile = open('%s-segment.csv' % logRoot, 'wb')
csvWriter = csv.writer(csvfile, dialect='excel')

# write table head
csvWriter.writerow(head)

# extract information of one iteration
f = open(metrics_filelist[0], 'r')
content = f.readlines()
f.close()

split_idx = [i for i in range(len(content)) if splitword in content[i]]
for i in range(split_idx[0], split_idx[1]):
    line = content[i]
    # print line
    if "[CUDA " in line:
        continue
    infor_line = line.split()
    infor_grid = " ".join(infor_line[2:5])
    infor_block = " ".join(infor_line[5:8])
    if infor_line[18] == 'void':
        infor_kernel = " ".join(infor_line[18:20])
    else:
        infor_kernel = infor_line[18]
    newkey = '.*'.join([infor_grid, infor_block, infor_kernel]) 
    # print newkey
    if newkey not in keywords:
        keywords.append(newkey)

keywords.append("[CUDA memcpy HtoD]")
keywords.append("[CUDA memcpy DtoH]")
keywords.append("[CUDA memset]")

for key in keywords:
    print key

for fp in metrics_filelist:
    # print fp

    baseInfo = fp.split('_')
    appName = baseInfo[1]
    coreF = str(int(baseInfo[2][4:]) + coreBase)
    memF = str(int(baseInfo[3][3:]) + memBase)

    # extract all kernels
    for keyword in keywords:

        rec = [appName, coreF, memF, keyword]
        print rec
        # extract execution time information
        f = open(fp, 'r')
        content = f.readlines()
        f.close()
        # arg = content[0][4:].strip()
        # rec.append(arg)

        # regex = re.compile(r'\d+\.\d+ (iter/s).')
        keyword = keyword.replace("(", "\(")
        keyword = keyword.replace(")", "\)")
        keyword = keyword.replace("[", "\[")
        keyword = keyword.replace("]", "\]")
        regex = re.compile(keyword)
        # print filter(regex.search, content)
        time = filter(regex.search, content)
        # time = [1.0 / float(line.split()[-7][1:]) for line in time]
        print len(time)
        # print time[0]
        rec.append(len(time))
        # if len(time) < 50:
        #     continue
        time = [line.split()[1] for line in time]

        for i in range(len(time)):
            if 'ms' in time[i]:
                time[i] = float(time[i][:-2]) / 1e3
            elif 'us' in time[i]:
                time[i] = float(time[i][:-2]) / 1e6
            elif 'ns' in time[i]:
                time[i] = float(time[i][:-2]) / 1e9
            else:
                time[i] = float(time[i][:-1])

        time.sort()
        # time = time[len(time)/3:len(time)*2/3]
        aver_time = np.mean(time)
        std_time = np.std(time)
        rec.append(str(aver_time))
        rec.append(str(std_time))
        rec.append("%f" % (std_time / aver_time))
        print rec

        csvWriter.writerow(rec)

