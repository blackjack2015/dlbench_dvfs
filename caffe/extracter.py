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

logRoot = 'titanX-googlenet-logs-v2'

perf_filelist = glob.glob(r'%s/*perf.log' % logRoot)
perf_filelist.sort()

head = ["appName", "coreF", "memF", "arg", "time/s", "power/W"]

coreBase = 1800
memBase = 4500
iters = 51

# prepare csv file
csvfile = open('%s.csv' % logRoot, 'wb')
csvWriter = csv.writer(csvfile, dialect='excel')

# write table head
csvWriter.writerow(head)

for fp in perf_filelist:
    # print fp

    baseInfo = fp.split('_')
    appName = baseInfo[1]
    coreF = str(int(baseInfo[2][4:]) + coreBase)
    memF = str(int(baseInfo[3][3:]) + memBase)

    rec = [appName, coreF, memF]

    # extract execution time information
    f = open(fp, 'r')
    content = f.readlines()
    f.close()
    arg = content[0][4:].strip()
    rec.append(arg)

    # regex = re.compile(r'\d+\.\d+ (iter/s).')
    regex = re.compile(r'(forward-backward)')
    # print filter(regex.search, content)
    time = filter(regex.search, content)
    # time = [1.0 / float(line.split()[-7][1:]) for line in time]
    time = [float(line.split()[-2])/1000.0 for line in time[:iters]]
    time.sort()
    aver_time = np.mean(time[len(time)/3:len(time)*2/3])
    rec.append(aver_time)

    # extract grid and block settings
    fm, number = re.subn('perf', 'power', fp)
    f = open(fm, 'r')
    content = f.readlines()
    maxLen = len(content)
    f.close()

    power = [float(line.split()[-1].strip()) / 1000 for line in content if line.split()[1].strip() == '2' or line.split()[1].strip() == '5']
    aver_power = np.mean(power[len(power)/3:len(power)*2/3])
    rec.append(aver_power)

    print rec
    csvWriter.writerow(rec[:len(head)])


# tempf = open('perfData.bin', 'wb')
# pickle.dump(record, tempf, 0)
# tempf.close()
