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

logRoot = 'logs'

perf_filelist = glob.glob(r'%s/*perf.log' % logRoot)
power_filelist = glob.glob(r'%s/*power.log' % logRoot)

perf_filelist.sort()
power_filelist.sort()

# Read GPU application settings
cf_ks = ConfigParser.SafeConfigParser()
cf_ks.read("dl_kernels_settings.cfg")
benchmark_programs = cf_ks.sections()

# num_input * num_output * w * h * kernel_w * kernel_h * batch_size
# (w, h, c, k, b, k_w, k_h, pad, stride) = (128, 128, 192, 64, 16, 3, 3, 0, 1)
cal_workload = 28991029248.0
head = ["appName", "coreF", "memF", "argNo", "time/us", "power/W", "GFLOPS", "GFLOPS/W"]
print head

# prepare csv file
csvfile = open('dvfs-conv.csv', 'wb')
csvWriter = csv.writer(csvfile, dialect='excel')

# write table head
csvWriter.writerow(head)

for fp in perf_filelist:
    # print fp

    baseInfo = fp.split('_')
    appName = baseInfo[1]+'-'+baseInfo[2]
    coreF = baseInfo[3][4:]
    memF = baseInfo[4][3:]
    argNo = baseInfo[5]

    rec = [appName, coreF, memF, argNo]

    # extract execution time information
    f = open(fp, 'r')
    content = f.readlines()
    f.close()
    message = content[-1].split()
    time = float(message[-2].strip())
    rec.append(time)

    # extract power information
    f = open(fp.replace('perf', 'power'), 'r')
    content = f.readlines()
    f.close()
    content = content[1:-1]
    powerSamples = content[len(content) / 2 - 20:len(content) / 2 + 20]
    powerSamples = [float(l.split()[-1].strip()) / 1000.0 for l in powerSamples]
    power = np.mean(powerSamples)
    rec.append(power)

    # append GFLOPS information
    rec.append(cal_workload / (time * 1000))

    # append energy efficiency
    rec.append(cal_workload / (time * 1000) / power)

    # print rec
    csvWriter.writerow(rec)

