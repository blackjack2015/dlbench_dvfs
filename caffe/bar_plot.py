import os 
import sys
import argparse

import xlrd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

DATA_PATH = 'excels'
csvfile = "titanX-vgg-logs-v2.csv"
METRIC = 'time/s'
metrics = ['coreF', 'memF', METRIC]
info = csvfile.split('-')
DEVICE = info[0]
NETWORK = info[1].upper()
OUTPUT_PATH = 'figures'

HATCH = ['//', '--', '\\\\', '||', '++', '--', '..', '++', '\\\\']
COLOR = ['#2F4F4F', '#808080', '#A9A9A9', '#778899', '#DCDCDC', '#556677', '#1D3E3E', '#808080', '#DCDCDC']

baseCF = 1600
baseMF = 3500

def plot_bars(dataF, metrics=['coreF', 'memF', 'time/s'], save_filename=None):

    print metrics
    dataF = dataF.sort_values(by=[metrics[0], metrics[1]])

    x1_axis = dataF[metrics[0]].unique()
    x2_axis = dataF[metrics[1]].unique()
    x1_axis.sort()
    x2_axis.sort()

    fig, ax = plt.subplots(figsize=(12,6))
    ax.grid(linestyle=':')
    bar_width = 0.9/len(x2_axis) 
    ind = np.arange(len(x1_axis)) + 0.05
    bars = []

    for i, omp in enumerate(x2_axis):
        print omp
        values = dataF[dataF[metrics[1]] == omp][metrics[2]]
        rects1 = ax.bar(ind, values, bar_width, color=COLOR[i%len(x2_axis)], hatch=HATCH[i%len(x2_axis)])
        bars.append(rects1)
        ind = ind + bar_width
        
    ax.set_ylabel(metrics[2], size='x-large')
    ax.set_xlabel('')
    margin = ax.get_ylim()[1]/4
    ax.set_ylim(top=ax.get_ylim()[1]+margin)

    num_compares = len(x2_axis)
    xticks = np.arange(len(x1_axis)) + bar_width*(num_compares / 2.0)
    ax.set_xticks(xticks)
    xlabels = tuple(["    %s %s" % (NETWORK, item) for item in x1_axis]) 
    ax.set_xticklabels(xlabels, size='medium', rotation=0)
    leg_ncol = len(x2_axis)
    if len(x2_axis) > 6:
        leg_ncol = 5
    ax.legend(tuple([bar[0] for bar in bars]), tuple(x2_axis), loc=9, ncol=leg_ncol, fontsize='x-large')
    if not save_filename:# or True:
        plt.show()
        return
    else:
        plt.savefig(os.path.join(OUTPUT_PATH, '%s.pdf'%save_filename), bbox_inches='tight')
        plt.savefig(os.path.join(OUTPUT_PATH, '%s.png'%save_filename), bbox_inches='tight')
    plt.clf()


df = pd.read_csv(os.path.join(DATA_PATH, csvfile), header = 0)
df["energy/J"] = df["time/s"] * df["power/W"]

print df.head(3)
print df.dtypes

curData = df

kCoreS = [ float((1 / item[METRIC] * curData[(curData['coreF'] == baseCF) & (curData['memF'] == item['memF'])][METRIC]) / \
          ((item['coreF']) * 1.0 / baseCF)) for idx, item in curData.iterrows() ]
print(kCoreS)
aver_coreS = np.array(kCoreS).prod() ** (1.0 / len(kCoreS))

# memory frequency sensitivity
kMemS = [ float((1 / item[METRIC] * curData[(curData['coreF'] == item['coreF']) & (curData['memF'] == baseMF)][METRIC]) / \
          ((item['memF']) * 1.0 / baseMF)) for idx, item in curData.iterrows() ]
print(kMemS)
aver_memS = np.array(kMemS).prod() ** (1.0 / len(kMemS))

# core-memory frequency sensitivity
kCMS = [ float((1 / item[METRIC] * curData[(curData['coreF'] == baseCF) & (curData['memF'] == baseMF)][METRIC]) / \
          (item['memF'] * 1.0 / baseMF * item['coreF'] * 1.0 / baseCF)) for idx, item in curData.iterrows() ]
print(kCMS)
aver_S = np.array(kCMS).prod() ** (1.0 / len(kCMS))

print('%s %s: %f %f %f' % (csvfile, METRIC, aver_coreS, aver_memS, aver_S))

# plot_bars(df, metrics, '%s-%s-%s' % (NETWORK, METRIC.split('/')[0], metrics[0]))
