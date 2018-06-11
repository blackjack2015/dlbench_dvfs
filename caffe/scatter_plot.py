import os 
import sys
import argparse

import xlrd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# from multiprocessing import Pool as ThreadPool
from multiprocessing.dummy import Pool as ThreadPool
from datetime import datetime

pool = ThreadPool(4)


DATA_PATH = 'excels'
csvfile = "titanX-googlenet-logs-v2-segment.csv"
info = csvfile.split('-')
DEVICE = info[0]
NETWORK = info[1].upper()
OUTPUT_PATH = 'figures'
baseCF = 1600
baseMF = 3500

HATCH = ['s', '8', '^', 'd','o']
COLOR = ['#FF9797', '#ADADAD', '#2894FF', '#28FF28', '#FFF3EE', '#FFBFFF', '#6C3365']
EXCLUDE = ['[CUDA memset]', '[CUDA memcpy DtoH]', '[CUDA memcpy HtoD]', \
           'caffe::sync_conv', 'caffe::sync_conv_groups(void)']

df = pd.read_csv(os.path.join(DATA_PATH, csvfile), header = 0)

df['short_kernel'] = None
for idx, item in df.iterrows():
    curKernel = df['kernel'][idx]
    if '[CUDA ' in curKernel:
        # df['short_kernel'][idx] = curKernel
        df.loc[idx, 'short_kernel'] = curKernel
        continue
    curKernel = curKernel.split('.*')[2]
    if 'void' in curKernel:
        # df['short_kernel'][idx] = curKernel.split('<')[0][4:] # remove 'void' and '<'
        df.loc[idx, 'short_kernel'] = curKernel.split('<')[0][4:] # remove 'void' and '<'
    else:
        df.loc[idx, 'short_kernel'] = curKernel

print df.head(3)
print df.dtypes


coreF = df['coreF'].unique()
memF = df['memF'].unique()
kernels = df['kernel'].unique()
short_kernels = df['short_kernel'].unique()

coreF.sort()
memF.sort()
kernels.sort()
short_kernels.sort()

coreS = []
breakdown = []
memS = []
cmS = []
# print "core sensitivity\tmemory sensitivity\tsensitivity\tkernel"


def kernelExtract(kernel):

    curData = df[df['kernel'] == kernel]
    # print curData

    # baseline kernel total time
    curBaseIdx = (curData['coreF'] == baseCF) & (curData['memF'] == baseMF)
    curBase = curData[curBaseIdx]
    curTime= sum(curBase['aver_time'] * curBase['count'])
    breakdown.append(curTime * 1.0 / baseTrain)

    # core frequency sensitivity
    kCoreS = [ float((item['aver_time'] / curData[(curData['coreF'] == baseCF) & (curData['memF'] == item['memF'])]['aver_time']) / \
              ((item['coreF']) * 1.0 / baseCF)) for idx, item in curData.iterrows() ]
    aver_coreS = np.mean(kCoreS)

    # memory frequency sensitivity
    kMemS = [ float((item['aver_time'] / curData[(curData['coreF'] == item['coreF']) & (curData['memF'] == baseMF)]['aver_time']) / \
              ((item['memF']) * 1.0 / baseMF)) for idx, item in curData.iterrows() ]
    aver_memS = np.mean(kMemS)

    # core-memory frequency sensitivity
    kCMS = [ float((item['aver_time'] / curData[(curData['coreF'] == baseCF) & (curData['memF'] == baseMF)]['aver_time']) / \
              (item['memF'] * 1.0 / baseMF * item['coreF'] * 1.0 / baseCF)) for idx, item in curData.iterrows() ]
    aver_S = np.mean(kCMS)

    # print "%f\t%f\t%f\t%s" % (aver_coreS, aver_memS, aver_S, kernel[:10])

    return aver_coreS, aver_memS, aver_S



# baseline total training time
baseTrain = df[(df['coreF'] == baseCF) & (df['memF'] == baseMF)]
baseTrain = sum(baseTrain['aver_time'] * baseTrain['count'])


print "start: ", datetime.now()

# # multi-threading
# results = pool.map(kernelExtract, kernels)
# pool.close()
# pool.join()

for kernel in kernels:
    curData = df[df['kernel'] == kernel]
    # print curData

    # baseline kernel total time
    curBaseIdx = (curData['coreF'] == baseCF) & (curData['memF'] == baseMF)
    curBase = curData[curBaseIdx]
    curTime= sum(curBase['aver_time'] * curBase['count'])
    breakdown.append(curTime * 1.0 / baseTrain)

    # core frequency sensitivity
    kCoreS = [ float((item['aver_time'] / curData[(curData['coreF'] == baseCF) & (curData['memF'] == item['memF'])]['aver_time']) / \
              ((item['coreF']) * 1.0 / baseCF)) for idx, item in curData.iterrows() ]
    aver_coreS = np.mean(kCoreS)

    # memory frequency sensitivity
    kMemS = [ float((item['aver_time'] / curData[(curData['coreF'] == item['coreF']) & (curData['memF'] == baseMF)]['aver_time']) / \
              ((item['memF']) * 1.0 / baseMF)) for idx, item in curData.iterrows() ]
    aver_memS = np.mean(kMemS)

    # core-memory frequency sensitivity
    kCMS = [ float((item['aver_time'] / curData[(curData['coreF'] == baseCF) & (curData['memF'] == baseMF)]['aver_time']) / \
              (item['memF'] * 1.0 / baseMF * item['coreF'] * 1.0 / baseCF)) for idx, item in curData.iterrows() ]
    aver_S = np.mean(kCMS)

    # print "%f\t%f\t%f\t%s" % (aver_coreS, aver_memS, aver_S, kernel[:10])

    coreS.append(aver_coreS)
    memS.append(aver_memS)
    cmS.append(aver_S)

print "end: ", datetime.now()

print len(coreS), len(memS), len(cmS), len(kernels), len(breakdown)
dataS = pd.DataFrame({'coreS': coreS, 'memS': memS, 'cmS': cmS, 'kernel': kernels, 'breakdown': breakdown})
# print dataS

# short labels
short_labels = []
word_length = 16
for i in range(len(kernels)):
    curKernel = kernels[i]
    if '[CUDA ' in curKernel:
        short_labels.append(curKernel)
        continue
    curKernel = curKernel.split('.*')[2]
    if curKernel.startswith( 'void' ):
        short_labels.append(curKernel.split('<')[0][4:4+word_length]) # remove 'void' and '<'
        # short_labels.append(curKernel.split('<')[0][4:]) # remove 'void' and '<'
    else:
        short_labels.append(curKernel[:word_length])
        # short_labels.append(curKernel)

# top 10 kernels
dataS = dataS[dataS['breakdown'] >= 0.010]
print len(dataS), sum(dataS['breakdown'])

# COLOR = np.random.rand(len(dataS)) * 50
# COLOR = np.arctan2(list(dataS['coreS']), list(dataS['memS']))
area = np.pi * (dataS['breakdown'] * 300) ** 2
labels = list(dataS['kernel'])


fig, ax = plt.subplots(figsize=(14,10))
scatters = []
legends = []
for i in range(len(labels)):
    if short_labels[i] in EXCLUDE:
        continue

    print i, short_labels[i], COLOR[i%len(COLOR)], HATCH[i%len(HATCH)]
    legends.append(short_labels[i])
    scatters.append(ax.scatter(list(dataS['coreS'])[i], list(dataS['memS'])[i], \
                            marker=HATCH[i%len(HATCH)], \
                            # alpha=0.5, \
                            c=COLOR[i%len(COLOR)], s=list(area)[i]))
    # ax.annotate(short_labels[i], (list(dataS['coreS'])[i], list(dataS['memS'])[i]))

lgnd = ax.legend(scatters, legends, scatterpoints=1, loc='lower left', ncol=4, fontsize=10)
for i in range(len(lgnd.legendHandles)):
    lgnd.legendHandles[i]._sizes = [100]

ax.set_ylabel('$S_{mem}^{PG}$', size='xx-large')
ax.set_xlabel('$S_{core}^{PG}$', size='xx-large')
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)

ax.grid(True)

# plt.show()
save_filename = '%s-%s' % (NETWORK, 'top-kernels')
plt.savefig(os.path.join(OUTPUT_PATH, '%s.pdf'%save_filename), bbox_inches='tight')
plt.savefig(os.path.join(OUTPUT_PATH, '%s.png'%save_filename), bbox_inches='tight')
# plt.clf()