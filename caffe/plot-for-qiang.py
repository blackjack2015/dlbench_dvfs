import os 
import sys
import argparse

import xlrd
import numpy as np
import matplotlib.pyplot as plt


DATA_PATH = '/mnt/hgfs/Dropbox/tmp'
OUTPUT_PATH = '/mnt/hgfs/Dropbox/tmp/qiang'

HATCH = ['//', '--', '\\\\', '||', '++', '--', '..', '++', '\\\\']
COLOR = ['#2F4F4F', '#808080', '#A9A9A9', '#778899', '#DCDCDC', '#556677', '#1D3E3E', '#808080', '#DCDCDC']
# SPECIAL_KEYS = ['E5-2630(energy)', 'E5-2630(time)', 'i7-3820(energy)', 'i7-3820(time)']
SPECIAL_KEYS = ['gtx980_dvfs(ops_per_watt)', 'EA(energy)', 'PA(time)', 'EE(ops_per_watt)']
LOG2_KEYS = ['i7-3820(time)', 'i7-3820(energy)', 'E5-2630(time)', 'E5-2630(energy)']
LOG10_KEYS = ['GPU(time)', 'GPU(energy)', 'EE(ops_per_watt)', 'EA(energy)', 'PA(time)']

def get_full_path_of_file(filename):
    return '%s/%s'%(DATA_PATH, filename)


def get_thoughtout_by_sheet(sheet, row, col):
    try:
        return sheet.row(row)[col].value
    except:
        return 0


def read_applications_from_sheet(sheet, metric='time'):
    """
    metric can be: time, power
    """
    nrows = sheet.nrows
    ncols = sheet.ncols
    nstart_row = 1
    applications = []
    metrics = {}
    compares = []
    for row in range(nstart_row, nrows):
        app = sheet.row(row)[0].value
        applications.append(app)
        metrics[app] = [] 
    iteration_col = 0
    for col in range(0, ncols):
        title = sheet.row(0)[col].value
        if title.find('Iterations') >= 0:
            iteration_col = col
            break

    for col in range(0, ncols):
        title = sheet.row(0)[col].value
        if title.find(metric) < 0:
            continue
        compares.append(title[0:-1].split('=')[1])
        for row in range(nstart_row, nrows):
            app = sheet.row(row)[0].value
            mvalue = float(get_thoughtout_by_sheet(sheet, row, col))
            #if metric == 'time':
                #mvalue = mvalue / int(get_thoughtout_by_sheet(sheet, row, iteration_col))
                #mvalue = np.log(mvalue)
            metrics[app].append(mvalue)
    return applications, metrics, compares


def plot_bars(sheet, metric='time', save_filename=None):
    apps, metrics, compares = read_applications_from_sheet(sheet, metric=metric)
    print apps, metrics
    print 'compares: ', compares

    if save_filename in SPECIAL_KEYS:
	fig, ax = plt.subplots(figsize=(16,6))
	fig.subplots_adjust(left=0.05, bottom=0.22,right=0.99, top=0.92, wspace=0.02, hspace=0.04)
    else:
   	fig, ax = plt.subplots()
    ax.grid(linestyle=':')
    bar_width = 0.9/len(compares) 
    ind = np.arange(len(apps))
    bars = []
#     if save_filename in LOG10_KEYS:
#         ax.set_yscale('log')
#     if save_filename in LOG2_KEYS:
#         ax.set_yscale('log')
    for i, omp in enumerate(compares):
        values = [metrics[app][i] for app in apps]
        if save_filename in LOG10_KEYS or save_filename in LOG2_KEYS:
	#     if save_filename in LOG2_KEYS:
	# 	ax.set_yscale('log', basey=2)
	#     else:
	# 	ax.set_yscale('log', basey=10)
            rects1 = ax.bar(ind, values, bar_width, color=COLOR[i%len(compares)], hatch=HATCH[i%len(compares)], log=True)
	else:
            rects1 = ax.bar(ind, values, bar_width, color=COLOR[i%len(compares)], hatch=HATCH[i%len(compares)])
        bars.append(rects1)
        ind = ind + bar_width
    if metric == 'time':
	if 'GPU' in save_filename:
	    ax.set_ylabel('Average Time (ms)', size='x-large')
	else:
            ax.set_ylabel('Average Time (s)', size='x-large')
    elif metric == 'power':
        ax.set_ylabel('Average Power (W)', size='x-large')
    elif metric == 'speedup':
        ax.set_ylabel('Speedup', size='x-large')
    elif metric == 'ops_per_watt':
        ax.set_ylabel('ops per watt', size='x-large')
    else:
        ax.set_ylabel('Energy (J)', size='x-large')
    # ax.set_xlabel('Application')
    ax.set_xlabel('')
    margin = ax.get_ylim()[1]/2.2
    ax.set_ylim(top=ax.get_ylim()[1]+margin)
    if save_filename in LOG10_KEYS or save_filename in LOG2_KEYS:
        ax.set_ylim(bottom=ax.get_ylim()[0]/10000, top=ax.get_ylim()[1]*100)
    else:    
	ax.set_ylim(top=ax.get_ylim()[1]+margin)
#     if save_filename in SPECIAL_KEYS:
# 	if not save_filename == 'EE(flops_per_watt)' and \
# 	   not save_filename == 'gtx980_dvfs(flops_per_watt)':
#             ax.set_ylim(top=7)
    num_compares = len(compares)
    xticks = np.arange(len(apps)) + bar_width*(num_compares / 2.0)
    ax.set_xticks(xticks)
    xlabels = tuple(apps) 
    ax.set_xticklabels(xlabels, size='x-large', rotation=90)
    leg_ncol = len(compares)
    if len(compares) > 6:
	leg_ncol = 5
    ax.legend(tuple([bar[0] for bar in bars]), tuple(compares), loc=9, ncol=leg_ncol, fontsize='x-large')
    if not save_filename:# or True:
        plt.show()
        return
    else:
        plt.savefig(os.path.join(OUTPUT_PATH, '%s.pdf'%save_filename), bbox_inches='tight')
        plt.savefig(os.path.join(OUTPUT_PATH, '%s.png'%save_filename), bbox_inches='tight')
    plt.clf()



def plot_from_file(filename):
    fullpath = get_full_path_of_file(filename)
    excel = xlrd.open_workbook(fullpath)
    sheet_names = excel.sheet_names()
    sheet_list = [excel.sheet_by_name(name) for name in sheet_names]
    for sheet_index in range(len(sheet_list)):
        #sheet_index = 1 
        filename = sheet_names[sheet_index]
        metric = sheet_names[sheet_index].split('(')[1][:-1]
	print metric
        sheet = sheet_list[sheet_index]
	print filename 
        plot_bars(sheet, metric=metric, save_filename=filename)
 
if __name__ == '__main__':
    plot_from_file('pber.xlsx')
