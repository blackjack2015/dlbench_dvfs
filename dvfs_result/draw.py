import os, sys
import argparse
import re
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
import numpy as np

parser = argparse.ArgumentParser(description='GPU measure')
parser.add_argument('-d', '--data-dir', default='.', type=str,
                    help='sssss')
args = parser.parse_args()
data_dir = args.data_dir
OUTPUT_PATH = 'pictures_out'

gpu = ['p100', 'v100', 'gtx2080ti']
algos = ['ipc_gemm', 'fft_tile', 'winograd']
nets = ['alexnet', 'resnet', 'vggnet', 'googlenet']
p100_coreF = ['544', '683', '810', '936', '1063', '1202', '1328']
v100_coreF = ['510', '652', '802', '945', '1087', '1237', '1380']
gtx2080ti_coreF = ['950', '1150', '1350', '1550', '1750', '2050']
gpu_coreFs = [p100_coreF, v100_coreF, gtx2080ti_coreF]
p100_memF = ['715']
v100_memF = ['877']
gtx2080ti_memF = ['5800', '6300', '6800', '7300']
gpu_memFs = [p100_memF, v100_memF, ['6300']]

gpu_index = {
    'p100': 0,
    'v100': 1,
    'gtx2080ti': 2    
}
batch_sizes = ['16', '32', '64', '128']
#==================================
auto_alexnet_batch_sizes = ['128', '256', '512', '1024']
auto_resnet_batch_sizes = ['16', '32']
auto_vggnet_batch_sizes = ['16', '32', '64', '128']
auto_googlenet_batch_sizes = ['16', '32', '64', '128']
auto_batch_sizes_list = [auto_alexnet_batch_sizes, auto_resnet_batch_sizes,
                        auto_vggnet_batch_sizes, auto_googlenet_batch_sizes]
#=====================================
ipc_gemm_alexnet_batch_sizes = ['128', '256', '512', '1024']
ipc_gemm_resnet_batch_sizes = ['16', '32']
ipc_gemm_vggnet_batch_sizes = ['16', '32', '64']
ipc_gemm_googlenet_batch_sizes = ['16', '32', '64', '128']
ipc_gemm_batch_sizes_list = [ipc_gemm_alexnet_batch_sizes, ipc_gemm_resnet_batch_sizes, 
                        ipc_gemm_vggnet_batch_sizes, ipc_gemm_googlenet_batch_sizes]
ipc_gemm_nets = ['alexnet', 'resnet', 'vggnet', 'googlenet']
#=======================================
winograd_nonfused_alexnet_batch_sizes = ['128', '256']
winograd_nonfused_resnet_batch_sizes = ['16', '32']
winograd_nonfused_vggnet_batch_sizes = ['16']
winograd_nonfused_googlenet_batch_sizes = ['16', '32', '64']
winograd_nonfused_batch_sizes_list = [winograd_nonfused_alexnet_batch_sizes, winograd_nonfused_resnet_batch_sizes, 
                        winograd_nonfused_vggnet_batch_sizes, winograd_nonfused_googlenet_batch_sizes]
winograd_nets = ['alexnet', 'resnet', 'vggnet', 'googlenet']
#========================================
fft_tile_alexnet_batch_sizes = ['128']
fft_tile_resnet_batch_sizes = [  ]
fft_tile_vggnet_batch_sizes = [  ]
fft_tile_googlenet_batch_sizes = ['16', '32']
fft_tile_batch_sizes_list = [fft_tile_alexnet_batch_sizes, fft_tile_googlenet_batch_sizes]
fft_tile_nets = ['alexnet', 'googlenet']
#=========================================



algo_nets = [ipc_gemm_nets, winograd_nets, fft_tile_nets]
algo_batch_sizes_list = [ipc_gemm_batch_sizes_list, winograd_nonfused_batch_sizes_list, fft_tile_batch_sizes_list]

variens_list = ['gpu', 'algo', 'framework', 'net', 'batch_size', 'coreF', 'memF']

algo_index = {
    'ipc_gemm': 0,
    'winograd': 1,
    'fft_tile': 2
}
#======================================
# plot_colors = ['b', 'g', 'r', 'k', 'c', 'm', 'y']
# plot_colors = ['#2a5caa', '#7fb80e', '#f58220', '#1d953f',
#             '#dea32c', '#00ae9d', '#f15a22', '#8552a1']
plot_colors = ['#66c2a5', '#fc8d62', '#8da0cb', '#a6d854',
            '#e78ac3', '#ffd92f', '#e5c494', '#b3b3b3']
markers = ['o', 'v', 's', '*']
linestyles = ['-', '--', '-.', ':']

size_ticks = 24
size_labels = 26

font_legend = {'family' : 'Times New Roman',
'weight' : 'normal',
'size' : 14,
}
font_ticks = {'family' : 'Times New Roman',
'weight' : 'normal',
'size' : 15,
}
font_labels = {'family' : 'Times New Roman',
'weight' : 'normal',
'size' : 26,
}

fig_x = 15
fig_y = 9
y_axis_interval = 10
y_max_ratio = 1.1

#======================================
# bar_colors = ['lightcoral', 'burlywood', 'y', 'yellowgreen',
#             'lightgreen', 'lightseagreen', 'lightskyblue', 'mediumpurple']
bar_total_width = 70

bar_colors = ['#0A64A4', '#24577B', '#03406A', '#3E94D1']
hatches = ['-', '+', 'x', '\\', '|', '/', 'O', '.']


def get_batch_time(gpu, algo, framework, net, batch_size, coreF, memF):
    gpu_dir = os.path.join(data_dir, gpu + '_' + algo + '_out')
    data_name = framework + '_' + net + '_' + batch_size + '_' + coreF + '_' + memF + '_perf.log'
    data_path = os.path.join(gpu_dir, data_name)
    try:
        data = open(data_path, 'r').readline().split()[0]
    except:
        data = 1
    return float(data)

def get_image_perf(gpu, algo, framework, net, batch_size, coreF, memF):
    gpu_dir = os.path.join(data_dir, gpu + '_' + algo + '_out')
    data_name = framework + '_' + net + '_' + batch_size + '_' + coreF + '_' + memF + '_perf.log'
    data_path = os.path.join(gpu_dir, data_name)
    try:
        data = open(data_path, 'r').readline().split()[1]
    except:
        data = 1
    return float(data)

def get_power(gpu, algo, framework, net, batch_size, coreF, memF):
    gpu_dir = os.path.join(data_dir, gpu + '_' + algo + '_out')
    data_name = framework + '_' + net + '_' + batch_size + '_' + coreF + '_' + memF + '_power.log'
    data_path = os.path.join(gpu_dir, data_name)
    # print(data_path)
    data = open(data_path, 'r').readline().split()[0]
    return float(data)

def get_energy(gpu, algo, framework, net, batch_size, coreF, memF):
    power = get_power(gpu, algo, framework, net, batch_size, coreF, memF)
    image_perf = get_image_perf(gpu, algo, framework, net, batch_size, coreF, memF)
    return (power / image_perf)

def get_power_respect(gpu=None, algo=None, framework=None, net=None, batch_size=None, coreF=None, memF=None):
    '''
        just one varient can be list
    '''
    powers = []
    varients = [gpu, algo, framework, net, batch_size, coreF, memF]
    for i, varient in enumerate(varients):
        if type(varient) == list:
            for item in varient:
                func_arg = []
                for j in range(len(varients)):
                    if i==j:
                        func_arg.append(item)
                    else:
                        func_arg.append(varients[j])

                powers.append(get_power(*func_arg))
            return powers


def get_batch_time_respect(gpu=None, algo=None, framework=None, net=None, batch_size=None, coreF=None, memF=None):
    '''
        just one varient can be list
    '''
    powers = []
    varients = [gpu, algo, framework, net, batch_size, coreF, memF]
    for i, varient in enumerate(varients):
        if type(varient) == list:
            for item in varient:
                func_arg = []
                for j in range(len(varients)):
                    if i==j:
                        func_arg.append(item)
                    else:
                        func_arg.append(varients[j])

                powers.append(get_batch_time(*func_arg))
            return powers



def get_image_perf_respect(gpu=None, algo=None, framework=None, net=None, batch_size=None, coreF=None, memF=None):
    '''
        just one varient can be list
    '''
    powers = []
    varients = [gpu, algo, framework, net, batch_size, coreF, memF]
    for i, varient in enumerate(varients):
        if type(varient) == list:
            for item in varient:
                func_arg = []
                for j in range(len(varients)):
                    if i==j:
                        func_arg.append(item)
                    else:
                        func_arg.append(varients[j])
                powers.append(get_image_perf(*func_arg))
            return powers

def get_energy_respect(gpu=None, algo=None, framework=None, net=None, batch_size=None, coreF=None, memF=None):
    powers = get_power_respect(gpu, algo, framework, net, batch_size, coreF, memF)
    image_perfs = get_image_perf_respect(gpu, algo, framework, net, batch_size, coreF, memF)
    energys = []
    for i in range(len(powers)):
        energys.append(powers[i] / image_perfs[i])
    return energys


# test_get_power = get_power('p100', 'auto', 'caffe', 'alexnet', '1024', '1063', '715')
# test_get_powers = get_power_respect('p100', 'auto', 'caffe', 'alexnet', '1024', p100_coreF, '715')
# test_get_powers_alex = get_power_respect('p100', 'auto', 'caffe', 'alexnet', alexnet_batch_sizes, '1063', '715')
# test_get_powers_resnet = get_power_respect('p100', 'auto', 'caffe', 'resnet', batch_sizes, '1063', '715')
# test_get_powers_google = get_power_respect('p100', 'auto', 'caffe', 'googlenet', batch_sizes, '1063', '715')
# test_get_image_perf_respect_batch = get_image_perf_respect('p100', 'auto', 'caffe', 'googlenet', batch_sizes, '1063', '715')
# test_get_energy_respect_batch = get_energy_respect('p100', 'auto', 'caffe', 'googlenet', batch_sizes, '1063', '715')

# print(test_get_power)
# print(test_get_powers)
# print(test_get_powers_alex)
# print(test_get_powers_resnet)
# print(test_get_powers_google)
# print(test_get_image_perf_respect_batch)
# print(test_get_energy_respect_batch)
# for coreF in p100_coreF:
#     print("=========***************============*************=============")
#     print("=========change coreF to  =%s ================================" % coreF)
#     print("=========***************============*************=============")
#     for net in ['vggnet', 'googlenet']:
#         test_get_powers_respect_batch = get_power_respect('p100', 'caffe', net, batch_sizes, coreF, '715')
#         test_get_image_perf_respect_batch = get_image_perf_respect('p100', 'caffe', net, batch_sizes, coreF, '715')
#         test_get_energy_respect_batch = get_energy_respect('p100', 'caffe', net, batch_sizes, coreF, '715')
#         print("=========batch change , print power, perf, energy =====%s ========================" % net)
#         print(test_get_powers_respect_batch)
#         print(test_get_image_perf_respect_batch)
#         print(test_get_energy_respect_batch)


#     test_get_powers_respect_batch = get_power_respect('p100', 'caffe', 'alexnet', alexnet_batch_sizes, coreF, '715')
#     test_get_image_perf_respect_batch = get_image_perf_respect('p100', 'caffe', 'alexnet', alexnet_batch_sizes, coreF, '715')
#     test_get_energy_respect_batch = get_energy_respect('p100', 'caffe', 'alexnet', alexnet_batch_sizes, coreF, '715')
#     print("=========batch change , print power, perf, energy =====%s ========================" % 'alexnet')
#     print(test_get_powers_respect_batch)
#     print(test_get_image_perf_respect_batch)
#     print(test_get_energy_respect_batch)


def draw_power_energy_fix_gpu_algo_net_config_batch_varient_frequency(
    gpu, algo, net, batch_sizes, gpu_coreF, gpu_memF, save=False):

    fig = plt.figure(figsize=(fig_x, fig_y))
    ax1 = fig.add_subplot(1, 1, 1)
    #==============
    ax2 = ax1.twinx()
    bar_i_width = 0

    bar_n = len(batch_sizes)
    bar_width = bar_total_width / bar_n
    np.array(gpu_coreF)
    # bar_x = np.array(gpu_coreF, np.dtype('int32')) - (bar_total_width - bar_width) / 2

    line_x = np.array([100, 200, 300, 400, 500, 600, 700], np.dtype('int32'))
    bar_x = line_x - (bar_total_width - bar_width) / 2

    #==============
    bars = []
    lines = []
    max_power = 0
    max_energy = 0
    for i_batch, batch in enumerate(batch_sizes):
        powers_respect_coreF = get_power_respect(gpu, algo, 'caffe', net, batch, gpu_coreF, gpu_memF)
        power_label_name = '{0}-b{1} Power'.format(net, batch)
        max_power = max(powers_respect_coreF) if max_power < max(powers_respect_coreF) else max_power
        bars.append(ax1.bar(bar_x + bar_i_width * bar_width, powers_respect_coreF, color=bar_colors[i_batch],
            width=bar_width, label=power_label_name))
        bar_i_width += 1

    for i_batch, batch in enumerate(batch_sizes):
        energy_respect_coreF = get_energy_respect(gpu, algo, 'caffe', net, batch, gpu_coreF, gpu_memF)       
        energy_label_name = '{0}-b{1} Energy'.format(net, batch)
        max_energy = max(energy_respect_coreF) if max_energy < max(energy_respect_coreF) else max_energy
        lines.append(ax2.plot(line_x, energy_respect_coreF, color=plot_colors[i_batch], marker=markers[i_batch],
            linewidth=2, label=energy_label_name))
    
    # y1_intervals = int(max_power/12)
    # y1_loc = plticker.MultipleLocator(base=y1_intervals)
    # ax1.yaxis.set_major_locator(y1_loc)
    # ax1.grid(True, linestyle="--", color="0.5", linewidth=1)

    box = ax1.get_position()
    ax1.set_position([box.x0, box.y0, box.width, box.height * 0.8])
    #ax1.legend(loc='upper center', bbox_to_anchor=(0., 0, 0.9, 0.1), ncol=4, prop=font_legend)
    #ax2.legend(loc='upper center', bbox_to_anchor=(0., 0, 0.92, 0.1), ncol=4, prop=font_legend)
    bars = list(bars)
    lines = sum(lines, [])
    legend_elements = []
    for i in range(len(bars)):
        legend_elements.append(bars[i])
        legend_elements.append(lines[i])

    print legend_elements
    ax1.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 1.14), ncol=len(bars), prop=font_legend)
    #ax2.legend(loc='upper center', ncol=4, prop=font_legend)

    ax1.set_yticks(np.linspace(0, max_power * y_max_ratio, num=y_axis_interval))
    ax2.set_yticks(np.linspace(0, max_energy * y_max_ratio, num=y_axis_interval))
    ax1.set_yticklabels(np.linspace(0, max_power * y_max_ratio, num=y_axis_interval).astype(np.int32), fontsize=size_ticks)
    ax2.set_yticklabels(np.linspace(0, max_energy * y_max_ratio, num=y_axis_interval).astype(np.int32), fontsize=size_ticks)
    ax1.grid(True, linestyle="--", color="0.5", linewidth=1)

    ax2.set_ylabel("%s (J/image)" % 'energy', size=size_labels)

    ax1.set_xlabel("%s (Hz)" % 'coreF', size=size_labels)
    ax1.set_ylabel("%s (J)" % 'power', size=size_labels)
    ax1.set_xticks(line_x)
    ax1.set_xticklabels([int(coreF) for coreF in gpu_coreF], size=size_ticks)

    # plt.xticks([int(coreF) for coreF in gpu_coreF], size=ticks_size)
    # plt.title("config [{0}, {1}], varient {2} ".format('net', 'batch', 'coreF'))
    if save == True:
        save_file_name = 'power&energy_ex1_{0}_{1}_{2}'.format(gpu, algo, net)
        plt.savefig(os.path.join(OUTPUT_PATH, '%s.pdf' % save_file_name), bbox_inches='tight')
        plt.savefig(os.path.join(OUTPUT_PATH, '%s.png' % save_file_name), bbox_inches='tight')
    else:
        plt.show()

def draw_power_energy_fix_gpu_algo_net_config_batch_varient_memfrequency(
    gpu, algo, net, batch_sizes, gpu_coreF, gpu_memF, save=False):

    fig = plt.figure(figsize=(fig_x, fig_y))
    ax1 = fig.add_subplot(1, 1, 1)
    #==============
    ax2 = ax1.twinx()
    bar_i_width = 0

    bar_n = len(batch_sizes)
    bar_width = bar_total_width / bar_n
    np.array(gpu_coreF)
    # bar_x = np.array(gpu_coreF, np.dtype('int32')) - (bar_total_width - bar_width) / 2

    line_x = np.array([100, 200, 300, 400, 500, 600, 700], np.dtype('int32'))
    bar_x = line_x - (bar_total_width - bar_width) / 2

    #==============
    max_power = 0
    max_energy = 0
    bars = []
    lines = []
    for i_batch, batch in enumerate(batch_sizes):
        powers_respect_coreF = get_power_respect(gpu, algo, 'caffe', net, batch, gpu_coreF, gpu_memF)
        power_label_name = '{0}-b{1} Power'.format(net, batch)
        max_power = max(powers_respect_coreF) if max_power < max(powers_respect_coreF) else max_power
        bars.append(ax1.bar(bar_x + bar_i_width * bar_width, powers_respect_coreF, color=bar_colors[i_batch],
            width=bar_width, label=power_label_name))
        bar_i_width += 1

    for i_batch, batch in enumerate(batch_sizes):
        energy_respect_coreF = get_energy_respect(gpu, algo, 'caffe', net, batch, gpu_coreF, gpu_memF)       
        energy_label_name = '{0}-b{1} Energy'.format(net, batch)
        max_energy = max(energy_respect_coreF) if max_energy < max(energy_respect_coreF) else max_energy
        lines.append(ax2.plot(line_x, energy_respect_coreF, color=plot_colors[i_batch], marker=markers[i_batch],
            linewidth=2, label=energy_label_name))
    
    # y1_intervals = int(max_power/12)
    # y1_loc = plticker.MultipleLocator(base=y1_intervals)
    # ax1.yaxis.set_major_locator(y1_loc)
    # ax1.grid(True, linestyle="--", color="0.5", linewidth=1)

    box = ax1.get_position()
    ax1.set_position([box.x0, box.y0, box.width, box.height * 0.8])

    bars = list(bars)
    lines = sum(lines, [])
    legend_elements = []
    for i in range(len(bars)):
        legend_elements.append(bars[i])
        legend_elements.append(lines[i])

    ax1.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 1.14), ncol=len(bars), prop=font_legend)
    #ax1.legend(loc='upper left', bbox_to_anchor=(0., 0.98, 0.8, 0.1), ncol=4, mode="expand", borderaxespad=0., prop=font_legend)
    #ax2.legend(loc='upper left', bbox_to_anchor=(0., 1.05, 0.8, 0.1), ncol=4, mode="expand", borderaxespad=0., prop=font_legend)

    ax1.set_yticks(np.linspace(0, max_power * y_max_ratio, num=y_axis_interval))
    ax2.set_yticks(np.linspace(0, max_energy * y_max_ratio, num=y_axis_interval))
    ax1.set_yticklabels(np.linspace(0, max_power * y_max_ratio, num=y_axis_interval).astype(np.int32), fontsize=size_ticks)
    ax2.set_yticklabels(np.linspace(0, max_energy * y_max_ratio, num=y_axis_interval).astype(np.int32), fontsize=size_ticks)
    ax1.grid(True, linestyle="--", color="0.5", linewidth=1)


    ax2.set_ylabel("%s (J/image)" % 'energy', size=size_labels)

    ax1.set_xlabel("%s (Hz)" % 'memF', size=size_labels)
    ax1.set_ylabel("%s (J)" % 'power', size=size_labels)
    ax1.set_xticks(line_x)
    ax1.set_xticklabels([int(coreF) for coreF in gpu_coreF], size=size_ticks)
    # plt.xticks([int(coreF) for coreF in gpu_memF])
    # plt.title("config [{0}, {1}], varient {2} ".format('net', 'batch', 'coreF'))
    if save == True:
        save_file_name = 'power&energy_ex1_{0}_{1}_{2}_memf'.format(gpu, algo, net)
        plt.savefig(os.path.join(OUTPUT_PATH, '%s.pdf' % save_file_name), bbox_inches='tight')
        plt.savefig(os.path.join(OUTPUT_PATH, '%s.png' % save_file_name), bbox_inches='tight')
    else:
        plt.show()


def draw_power_energy_ratio_fix_gpu_algo_net_config_batch_varient_frequency(
    gpu, algo, net, batch_sizes, gpu_coreF, gpu_memF):

    fig = plt.figure(figsize=(fig_x, fig_y))
    ax1 = fig.add_subplot(1, 1, 1)
    #==============
    ax2 = ax1.twinx()
    bar_i_width = 0

    bar_n = len(batch_sizes)
    bar_width = bar_total_width / bar_n
    np.array(gpu_coreF)
    # bar_x = np.array(gpu_coreF, np.dtype('int32')) - (bar_total_width - bar_width) / 2

    line_x = np.array([100, 200, 300, 400, 500, 600, 700], np.dtype('int32'))
    bar_x = line_x - (bar_total_width - bar_width) / 2

    #==============
    #==============
    max_power = 0
    max_energy = 0
    bars = []
    lines = []
    for i_batch, batch in enumerate(batch_sizes):
        base_power = get_power(gpu, algo, 'caffe', net, batch, gpu_coreF[0], gpu_memF)
        powers_respect_coreF = get_power_respect(gpu, algo, 'caffe', net, batch, gpu_coreF, gpu_memF)
        powers_respect_coreF_ration = np.array(powers_respect_coreF, np.dtype('float')) / float(base_power)

        max_power = np.max(powers_respect_coreF_ration) if max_power < np.max(powers_respect_coreF_ration) else max_power
        power_label_name = '{0}-b{1} Power'.format(net, batch)
        bars.append(ax1.bar(bar_x + bar_i_width * bar_width, powers_respect_coreF_ration, color=bar_colors[i_batch],
            width=bar_width, label=power_label_name))
        bar_i_width += 1

    for i_batch, batch in enumerate(batch_sizes):
        base_energy = get_energy(gpu, algo, 'caffe', net, batch, gpu_coreF[0], gpu_memF)
        energy_respect_coreF = get_energy_respect(gpu, algo, 'caffe', net, batch, gpu_coreF, gpu_memF)
        energy_respect_coreF_ration = np.array(energy_respect_coreF, np.dtype('float')) / float(base_energy)

        max_energy = np.max(energy_respect_coreF_ration) if max_energy < np.max(energy_respect_coreF_ration) else max_energy
        energy_label_name = '{0}-b{1} Energy'.format(net, batch)
        lines.append(ax2.plot(line_x, energy_respect_coreF_ration, color=plot_colors[i_batch], marker=markers[i_batch],
            linewidth=2, label=energy_label_name))
    
    # y1_intervals = float(max_power/12)
    # y1_loc = plticker.MultipleLocator(base=y1_intervals)
    # ax1.yaxis.set_major_locator(y1_loc)
    # ax1.grid(True, linestyle="--", color="0.5", linewidth=1)

    box = ax1.get_position()
    ax1.set_position([box.x0, box.y0, box.width, box.height * 0.8])

    bars = list(bars)
    lines = sum(lines, [])
    legend_elements = []
    for i in range(len(bars)):
        legend_elements.append(bars[i])
        legend_elements.append(lines[i])

    ax1.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 1.14), ncol=len(bars), prop=font_legend)
    #ax1.legend(loc='upper left', bbox_to_anchor=(0., 0.98, 0.8, 0.1), ncol=4, mode="expand", borderaxespad=0., prop=font_legend)
    #ax2.legend(loc='upper left', bbox_to_anchor=(0., 1.05, 0.8, 0.1), ncol=4, mode="expand", borderaxespad=0., prop=font_legend)
    
    ax1.set_yticks(np.linspace(0, max_power * y_max_ratio, num=y_axis_interval))
    ax2.set_yticks(np.linspace(0, energy_respect_coreF * y_max_ratio, num=y_axis_interval))
    ax1.set_yticklabels(np.linspace(0, max_power * y_max_ratio, num=y_axis_interval).astype(np.int32), fontsize=size_ticks)
    ax2.set_yticklabels(np.linspace(0, energy_respect_coreF * y_max_ratio, num=y_axis_interval).astype(np.int32), fontsize=size_ticks)
    ax1.grid(True, linestyle="--", color="0.5", linewidth=1)

    ax2.set_ylabel("%s (J/image)" % 'energy', size=size_labels)

    ax1.set_xlabel("%s (Hz)" % 'coreF', size=size_labels)
    ax1.set_ylabel("%s (J)" % 'power', size=size_labels)
    ax1.set_xticks(line_x)
    ax1.set_xticklabels([int(coreF) for coreF in gpu_coreF], size=size_ticks)
    # plt.xticks([int(coreF) for coreF in gpu_coreF])
    # plt.title("config [{0}, {1}], varient {2} ".format('net', 'batch', 'coreF'))


def draw_power_perf_fix_gpu_algo_net_config_batch_varient_frequency(
    gpu, algo, net, batch_sizes, gpu_coreF, gpu_memF, save=False):

    fig = plt.figure(figsize=(fig_x, fig_y))
    ax1 = fig.add_subplot(1, 1, 1)
    #==============
    ax2 = ax1.twinx()
    bar_i_width = 0

    bar_n = len(batch_sizes)
    bar_width = bar_total_width / bar_n
    np.array(gpu_coreF)
    # bar_x = np.array(gpu_coreF, np.dtype('int32')) - (bar_total_width - bar_width) / 2

    line_x = np.array([100, 200, 300, 400, 500, 600, 700], np.dtype('int32'))
    bar_x = line_x - (bar_total_width - bar_width) / 2

    #==============
    #==============
    max_power = 0
    max_image_perf = 0
    bars = []
    lines = []
    for i_batch, batch in enumerate(batch_sizes):
        powers_respect_coreF = get_power_respect(gpu, algo, 'caffe', net, batch, gpu_coreF, gpu_memF)
        power_label_name = '{0}-b{1} Power'.format(net, batch)
        max_power = max(powers_respect_coreF) if max_power < max(powers_respect_coreF) else max_power
        bars.append(ax1.bar(bar_x + bar_i_width * bar_width, powers_respect_coreF, color=bar_colors[i_batch],
            width=bar_width, label=power_label_name))
        bar_i_width += 1

    for i_batch, batch in enumerate(batch_sizes):
        image_perf_respect_coreF = get_image_perf_respect(gpu, algo, 'caffe', net, batch, gpu_coreF, gpu_memF)       
        max_image_perf = max(image_perf_respect_coreF) if max_image_perf < max(image_perf_respect_coreF) else max_image_perf

        energy_label_name = '{0}-b{1} image_per'.format(net, batch)
        lines.append(ax2.plot(line_x, image_perf_respect_coreF, color=plot_colors[i_batch], marker=markers[i_batch],
            linewidth=2, label=energy_label_name))
    
    # y1_intervals = int(max_power/12)
    # y1_loc = plticker.MultipleLocator(base=y1_intervals)
    # ax1.yaxis.set_major_locator(y1_loc)
    # ax1.grid(True, linestyle="--", color="0.5", linewidth=1)

    box = ax1.get_position()
    ax1.set_position([box.x0, box.y0, box.width, box.height * 0.8])

    bars = list(bars)
    lines = sum(lines, [])
    legend_elements = []
    for i in range(len(bars)):
        legend_elements.append(bars[i])
        legend_elements.append(lines[i])

    ax1.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 1.14), ncol=len(bars), prop=font_legend)
    #ax1.legend(loc='upper left', bbox_to_anchor=(0., 0.98, 0.8, 0.1), ncol=4, mode="expand", borderaxespad=0., prop=font_legend)
    #ax2.legend(loc='upper left', bbox_to_anchor=(0., 1.05, 0.8, 0.1), ncol=4, mode="expand", borderaxespad=0., prop=font_legend)
    
    ax1.set_yticks(np.linspace(0, max_power * y_max_ratio, num=y_axis_interval))
    ax2.set_yticks(np.linspace(0, max_image_perf * y_max_ratio, num=y_axis_interval))
    ax1.set_yticklabels(np.linspace(0, max_power * y_max_ratio, num=y_axis_interval).astype(np.int32), fontsize=size_ticks)
    ax2.set_yticklabels(np.linspace(0, max_image_perf * y_max_ratio, num=y_axis_interval).astype(np.int32), fontsize=size_ticks)
    ax1.grid(True, linestyle="--", color="0.5", linewidth=1)

    ax2.set_ylabel("%s (image/s)" % 'perf', size=size_labels)
    ax1.set_xlabel("%s (Hz)" % 'coreF', size=size_labels)
    ax1.set_ylabel("%s (J)" % 'power', size=size_labels)
    ax1.set_xticks(line_x)
    ax1.set_xticklabels([int(coreF) for coreF in gpu_coreF], size=size_ticks)
    # plt.xticks([int(coreF) for coreF in gpu_coreF])
    # plt.title("config [{0}, {1}], varient {2} ".format('net', 'batch', 'coreF'))
    if save == True:
        save_file_name = 'power&perf_ex1_{0}_{1}_{2}'.format(gpu, algo, net)
        plt.savefig(os.path.join(OUTPUT_PATH, '%s.pdf' % save_file_name), bbox_inches='tight')
        plt.savefig(os.path.join(OUTPUT_PATH, '%s.png' % save_file_name), bbox_inches='tight')
    else:
        plt.show()

def draw_power_perf_fix_gpu_algo_net_config_batch_varient_memfrequency(
    gpu, algo, net, batch_sizes, gpu_coreF, gpu_memF, save=False):

    fig = plt.figure(figsize=(fig_x, fig_y))
    ax1 = fig.add_subplot(1, 1, 1)
    #==============
    ax2 = ax1.twinx()
    bar_i_width = 0

    bar_n = len(batch_sizes)
    bar_width = bar_total_width / bar_n
    np.array(gpu_coreF)
    # bar_x = np.array(gpu_coreF, np.dtype('int32')) - (bar_total_width - bar_width) / 2

    line_x = np.array([100, 200, 300, 400, 500, 600, 700], np.dtype('int32'))
    bar_x = line_x - (bar_total_width - bar_width) / 2

    #==============
    #==============
    max_power = 0
    max_image_perf = 0
    bars = []
    lines = []
    for i_batch, batch in enumerate(batch_sizes):
        powers_respect_coreF = get_power_respect(gpu, algo, 'caffe', net, batch, gpu_coreF, gpu_memF)
        power_label_name = '{0}-b{1} Power'.format(net, batch)
        max_power = max(powers_respect_coreF) if max_power < max(powers_respect_coreF) else max_power
        bars.append(ax1.bar(bar_x + bar_i_width * bar_width, powers_respect_coreF, color=bar_colors[i_batch],
            width=bar_width, label=power_label_name))
        bar_i_width += 1

    for i_batch, batch in enumerate(batch_sizes):
        image_perf_respect_coreF = get_image_perf_respect(gpu, algo, 'caffe', net, batch, gpu_coreF, gpu_memF)       
        max_image_perf = max(image_perf_respect_coreF) if max_image_perf < max(image_perf_respect_coreF) else max_image_perf

        energy_label_name = '{0}-b{1} image_per'.format(net, batch)
        bars.append(ax2.plot(line_x, image_perf_respect_coreF, color=plot_colors[i_batch], marker=markers[i_batch],
            linewidth=2, label=energy_label_name))
    
    # y1_intervals = int(max_power/12)
    # y1_loc = plticker.MultipleLocator(base=y1_intervals)
    # ax1.yaxis.set_major_locator(y1_loc)
    # ax1.grid(True, linestyle="--", color="0.5", linewidth=1)

    box = ax1.get_position()
    ax1.set_position([box.x0, box.y0, box.width, box.height * 0.8])

    bars = list(bars)
    lines = sum(lines, [])
    legend_elements = []
    for i in range(len(bars)):
        legend_elements.append(bars[i])
        legend_elements.append(lines[i])

    ax1.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 1.14), ncol=len(bars), prop=font_legend)
    #ax1.legend(loc='upper left', bbox_to_anchor=(0., 0.98, 0.8, 0.1), ncol=4, mode="expand", borderaxespad=0., prop=font_legend)
    #ax2.legend(loc='upper left', bbox_to_anchor=(0., 1.05, 0.8, 0.1), ncol=4, mode="expand", borderaxespad=0., prop=font_legend)
    # ax2.set_ylim([0, max_image_perf*1.1])

    ax2.set_ylabel("%s (image/s)" % 'perf', size=size_labels)

    ax1.set_yticks(np.linspace(0, max_power * y_max_ratio, num=y_axis_interval))
    ax2.set_yticks(np.linspace(0, max_image_perf * y_max_ratio, num=y_axis_interval))
    ax1.set_yticklabels(np.linspace(0, max_power * y_max_ratio, num=y_axis_interval).astype(np.int32), fontsize=size_ticks)
    ax2.set_yticklabels(np.linspace(0, max_image_perf * y_max_ratio, num=y_axis_interval).astype(np.int32), fontsize=size_ticks)
    ax1.grid(True, linestyle="--", color="0.5", linewidth=1)

    ax1.set_xlabel("%s (Hz)" % 'memF', size=size_labels)
    ax1.set_ylabel("%s (J)" % 'power', size=size_labels)
    ax1.set_xticks(line_x)
    ax1.set_xticklabels([int(coreF) for coreF in gpu_coreF], size=size_ticks)
    # plt.xticks([int(coreF) for coreF in gpu_memF])
    # plt.title("config [{0}, {1}], varient {2} ".format('net', 'batch', 'coreF'))
    if save == True:
        save_file_name = 'power&perf_ex1_{0}_{1}_{2}_memF'.format(gpu, algo, net)
        plt.savefig(os.path.join(OUTPUT_PATH, '%s.pdf' % save_file_name), bbox_inches='tight')
        plt.savefig(os.path.join(OUTPUT_PATH, '%s.png' % save_file_name), bbox_inches='tight')
    else:
        plt.show()

def draw_power_perf_ratio_fix_gpu_algo_net_config_batch_varient_frequency(
    gpu, algo, net, batch_sizes, gpu_coreF, gpu_memF):

    fig = plt.figure(figsize=(fig_x, fig_y))
    ax1 = fig.add_subplot(1, 1, 1)
    #==============
    ax2 = ax1.twinx()
    bar_i_width = 0

    bar_n = len(batch_sizes)
    bar_width = bar_total_width / bar_n
    np.array(gpu_coreF)
    # bar_x = np.array(gpu_coreF, np.dtype('int32')) - (bar_total_width - bar_width) / 2

    line_x = np.array([100, 200, 300, 400, 500, 600, 700], np.dtype('int32'))
    bar_x = line_x - (bar_total_width - bar_width) / 2

    #==============
    #==============
    max_power = 0
    max_image_perf = 0

    bars = []
    lines = []
    for i_batch, batch in enumerate(batch_sizes):
        base_power = get_power(gpu, algo, 'caffe', net, batch, gpu_coreF[0], gpu_memF)
        powers_respect_coreF = get_power_respect(gpu, algo, 'caffe', net, batch, gpu_coreF, gpu_memF)
        powers_respect_coreF_ration = np.array(powers_respect_coreF, np.dtype('float')) / float(base_power)


        max_power = np.max(powers_respect_coreF_ration) if max_power < np.max(powers_respect_coreF_ration) else max_power
        power_label_name = '{0}-b{1} Power'.format(net, batch)
        bars.append(ax1.bar(bar_x + bar_i_width * bar_width, powers_respect_coreF_ration, color=bar_colors[i_batch],
            width=bar_width, label=power_label_name))
        bar_i_width += 1

    for i_batch, batch in enumerate(batch_sizes):
        base_image_perf = get_image_perf(gpu, algo, 'caffe', net, batch, gpu_coreF[0], gpu_memF)
        image_perf_respect_coreF = get_image_perf_respect(gpu, algo, 'caffe', net, batch, gpu_coreF, gpu_memF)       
        image_perf_respect_coreF_ration = np.array(image_perf_respect_coreF, np.dtype('float')) / float(base_image_perf)

        max_image_perf = np.max(image_perf_respect_coreF_ration) if max_image_perf < np.max(image_perf_respect_coreF_ration) else max_image_perf

        energy_label_name = '{0}-b{1} image_per'.format(net, batch)
        lines.append(ax2.plot(line_x, image_perf_respect_coreF_ration, color=plot_colors[i_batch], marker=markers[i_batch],
            linewidth=2, label=energy_label_name))
    
    # y1_intervals = float(max_power/12)
    # y1_loc = plticker.MultipleLocator(base=y1_intervals)
    # ax1.yaxis.set_major_locator(y1_loc)
    # ax1.grid(True, linestyle="--", color="0.5", linewidth=1)

    box = ax1.get_position()
    ax1.set_position([box.x0, box.y0, box.width, box.height * 0.8])

    bars = list(bars)
    lines = sum(lines, [])
    legend_elements = []
    for i in range(len(bars)):
        legend_elements.append(bars[i])
        legend_elements.append(lines[i])

    ax1.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 1.14), ncol=len(bars), prop=font_legend)


    #ax1.legend(loc='upper left', bbox_to_anchor=(0., 0.98, 0.8, 0.1), ncol=4, mode="expand", borderaxespad=0., prop=font_legend)

    #ax2.legend(loc='upper left', bbox_to_anchor=(0., 1.05, 0.8, 0.1), ncol=4, mode="expand", borderaxespad=0., prop=font_legend)
    # ax2.set_ylim([0, max_image_perf*1.1])
    ax2.set_ylabel("%s (image/s)" % 'perf', size=size_labels)
    
    ax1.set_yticks(np.linspace(0, max_power * y_max_ratio, num=y_axis_interval))
    ax2.set_yticks(np.linspace(0, max_image_perf * y_max_ratio, num=y_axis_interval))
    ax1.set_yticklabels(np.linspace(0, max_power * y_max_ratio, num=y_axis_interval).astype(np.int32), fontsize=size_ticks)
    ax2.set_yticklabels(np.linspace(0, max_image_perf * y_max_ratio, num=y_axis_interval).astype(np.int32), fontsize=size_ticks)
    ax1.grid(True, linestyle="--", color="0.5", linewidth=1)

    ax1.set_xlabel("%s (Hz)" % 'coreF', size=size_labels)
    ax1.set_ylabel("%s (J)" % 'power', size=size_labels)
    ax1.set_xticks(line_x)
    ax1.set_xticklabels([int(coreF) for coreF in gpu_coreF], size=size_ticks)
    # plt.xticks([int(coreF) for coreF in gpu_coreF])
    # plt.title("config [{0}, {1}], varient {2} ".format('net', 'batch', 'coreF'))

# auto_alexnet_batch_sizes = ['128', '256', '512', '1024']
# auto_resnet_batch_sizes = ['16', '32']
# auto_vggnet_batch_sizes = ['16', '32', '64']
# auto_googlenet_batch_sizes = ['16', '32', '64', '128']
# auto_batch_sizes_list = [auto_alexnet_batch_sizes, auto_resnet_batch_sizes, 
#                         auto_vggnet_batch_sizes, auto_googlenet_batch_sizes]

# for i_net, net in enumerate(nets):

#     draw_power_energy_fix_gpu_algo_net_config_batch_varient_frequency(
#         'p100', 'winograd_nonfused', net, winograd_nonfused_batch_sizes_list[i_net], p100_coreF, p100_memF[0])
#     draw_power_perf_fix_gpu_algo_net_config_batch_varient_frequency(
#         'p100', 'winograd_nonfused', net, winograd_nonfused_batch_sizes_list[i_net], p100_coreF, p100_memF[0])















ipc_gemm_alexnet_batch_sizes = ['128', '256', '512', '1024']
ipc_gemm_resnet_batch_sizes = ['16', '32']
ipc_gemm_vggnet_batch_sizes = ['16', '32', '64']
ipc_gemm_googlenet_batch_sizes = ['16', '32', '64', '128']
ipc_gemm_batch_sizes_list = [ipc_gemm_alexnet_batch_sizes, ipc_gemm_resnet_batch_sizes, 
                        ipc_gemm_vggnet_batch_sizes, ipc_gemm_googlenet_batch_sizes]

# for i_net, net in enumerate(nets):

#     draw_power_energy_fix_gpu_algo_net_config_batch_varient_frequency(
#         'v100', 'auto', net, auto_batch_sizes_list[i_net], v100_coreF, v100_memF[0])
#     draw_power_perf_fix_gpu_algo_net_config_batch_varient_frequency(
#         'v100', 'auto', net, auto_batch_sizes_list[i_net], v100_coreF, v100_memF[0])

for i_net, net in enumerate(nets):
    draw_power_energy_fix_gpu_algo_net_config_batch_varient_frequency(
        'v100', 'ipc_gemm', net, ipc_gemm_batch_sizes_list[i_net], v100_coreF, v100_memF[0])
    draw_power_perf_fix_gpu_algo_net_config_batch_varient_frequency(
        'v100', 'ipc_gemm', net, ipc_gemm_batch_sizes_list[i_net], v100_coreF, v100_memF[0])

# draw_power_energy_fix_gpu_algo_net_config_batch_varient_frequency(
#     'v100', 'winograd', 'alexnet', winograd_nonfused_alexnet_batch_sizes, v100_coreF, v100_memF[0], save=True)

# for i_net, net in enumerate(nets):
#     draw_power_energy_fix_gpu_algo_net_config_batch_varient_frequency(
#         'v100', 'winograd', net, winograd_nonfused_batch_sizes_list[i_net], v100_coreF, v100_memF[0], save=True)
#     draw_power_perf_fix_gpu_algo_net_config_batch_varient_frequency(
#         'v100', 'winograd', net, winograd_nonfused_batch_sizes_list[i_net], v100_coreF, v100_memF[0], save=True)

# for i_net, net in enumerate(fft_tile_nets):
#     draw_power_energy_fix_gpu_algo_net_config_batch_varient_frequency(
#         'v100', 'fft_tile', net, fft_tile_batch_sizes_list[i_net], v100_coreF, v100_memF[0], save=True)
#     draw_power_perf_fix_gpu_algo_net_config_batch_varient_frequency(
#         'v100', 'fft_tile', net, fft_tile_batch_sizes_list[i_net], v100_coreF, v100_memF[0], save=True)

















# for i_net, net in enumerate(nets):

#     draw_power_energy_fix_gpu_algo_net_config_batch_varient_frequency(
#         'p100', 'auto', net, auto_batch_sizes_list[i_net], p100_coreF, p100_memF[0], save=True)
#     draw_power_perf_fix_gpu_algo_net_config_batch_varient_frequency(
#         'p100', 'auto', net, auto_batch_sizes_list[i_net], p100_coreF, p100_memF[0], save=True)

# for i_net, net in enumerate(nets):

#     draw_power_energy_fix_gpu_algo_net_config_batch_varient_frequency(
#         'p100', 'ipc_gemm', net, ipc_gemm_batch_sizes_list[i_net], p100_coreF, p100_memF[0])
#     draw_power_perf_fix_gpu_algo_net_config_batch_varient_frequency(
#         'p100', 'ipc_gemm', net, ipc_gemm_batch_sizes_list[i_net], p100_coreF, p100_memF[0])

# for i_net, net in enumerate(nets):
#     draw_power_energy_fix_gpu_algo_net_config_batch_varient_frequency(
#         'p100', 'winograd_nonfused', net, winograd_nonfused_batch_sizes_list[i_net], p100_coreF, p100_memF[0])
#     draw_power_perf_fix_gpu_algo_net_config_batch_varient_frequency(
#         'p100', 'winograd_nonfused', net, winograd_nonfused_batch_sizes_list[i_net], p100_coreF, p100_memF[0])

# for i_net, net in enumerate(fft_tile_nets):
#     draw_power_energy_fix_gpu_algo_net_config_batch_varient_frequency(
#         'p100', 'fft_tile', net, fft_tile_batch_sizes_list[i_net], p100_coreF, p100_memF[0], save=True)
#     draw_power_perf_fix_gpu_algo_net_config_batch_varient_frequency(
#         'p100', 'fft_tile', net, fft_tile_batch_sizes_list[i_net], p100_coreF, p100_memF[0], save=True)













gtx2080ti_ipc_gemm_alexnet_batch_sizes = ['64', '128', '256', '512']
gtx2080ti_ipc_gemm_resnet_batch_sizes = ['8', '16']
gtx2080ti_ipc_gemm_vggnet_batch_sizes = ['16', '32', '64']
gtx2080ti_ipc_gemm_googlenet_batch_sizes = ['16', '32', '64', '128']
gtx2080ti_ipc_gemm_batch_sizes_list = [gtx2080ti_ipc_gemm_alexnet_batch_sizes, gtx2080ti_ipc_gemm_resnet_batch_sizes, 
                        gtx2080ti_ipc_gemm_vggnet_batch_sizes, gtx2080ti_ipc_gemm_googlenet_batch_sizes]
# ======================================core F of gtx2080ti
# for i_net, net in enumerate(nets):

#     draw_power_perf_ratio_fix_gpu_algo_net_config_batch_varient_frequency(
#         'gtx2080ti', 'auto', net, gtx2080ti_ipc_gemm_batch_sizes_list[i_net], gtx2080ti_coreF, gtx2080ti_memF[1])
#     draw_power_perf_fix_gpu_algo_net_config_batch_varient_frequency(
#         'gtx2080ti', 'auto', net, gtx2080ti_ipc_gemm_batch_sizes_list[i_net], gtx2080ti_coreF, gtx2080ti_memF[1])

# for i_net, net in enumerate(nets):

#     draw_power_perf_fix_gpu_algo_net_config_batch_varient_frequency(
#         'gtx2080ti', 'ipc_gemm', net, gtx2080ti_ipc_gemm_batch_sizes_list[i_net], gtx2080ti_coreF, gtx2080ti_memF[1])
#     draw_power_energy_fix_gpu_algo_net_config_batch_varient_frequency(
#         'gtx2080ti', 'ipc_gemm', net, gtx2080ti_ipc_gemm_batch_sizes_list[i_net], gtx2080ti_coreF, gtx2080ti_memF[1])

# for i_net, net in enumerate(nets):
#     draw_power_energy_fix_gpu_algo_net_config_batch_varient_frequency(
#         'gtx2080ti', 'winograd_nonfused', net, winograd_nonfused_batch_sizes_list[i_net], gtx2080ti_coreF, gtx2080ti_memF[1])
#     draw_power_perf_fix_gpu_algo_net_config_batch_varient_frequency(
#         'gtx2080ti', 'winograd_nonfused', net, winograd_nonfused_batch_sizes_list[i_net], gtx2080ti_coreF, gtx2080ti_memF[1])

# for i_net, net in enumerate(nets):
#     draw_power_energy_fix_gpu_algo_net_config_batch_varient_frequency(
#         'gtx2080ti', 'fft_tile', net, fft_tile_batch_sizes_list[i_net], gtx2080ti_coreF, gtx2080ti_memF[1])
#     draw_power_perf_fix_gpu_algo_net_config_batch_varient_frequency(
#         'gtx2080ti', 'fft_tile', net, fft_tile_batch_sizes_list[i_net], gtx2080ti_coreF, gtx2080ti_memF[1])

# ======================================mem F of gtx2080ti
# for i_net, net in enumerate(nets):

#     draw_power_perf_ratio_fix_gpu_algo_net_config_batch_varient_frequency(
#         'gtx2080ti', 'auto', net, auto_batch_sizes_list[i_net], gtx2080ti_coreF, gtx2080ti_memF[1])
#     draw_power_perf_fix_gpu_algo_net_config_batch_varient_frequency(
#         'gtx2080ti', 'auto', net, auto_batch_sizes_list[i_net], gtx2080ti_coreF, gtx2080ti_memF[1])

# for i_net, net in enumerate(nets):

#     draw_power_perf_fix_gpu_algo_net_config_batch_varient_memfrequency(
#         'gtx2080ti', 'ipc_gemm', net, gtx2080ti_ipc_gemm_batch_sizes_list[i_net], gtx2080ti_coreF[3], gtx2080ti_memF)
#     draw_power_energy_fix_gpu_algo_net_config_batch_varient_memfrequency(
#         'gtx2080ti', 'ipc_gemm', net, gtx2080ti_ipc_gemm_batch_sizes_list[i_net], gtx2080ti_coreF[3], gtx2080ti_memF)

# for i_net, net in enumerate(nets):
#     draw_power_energy_fix_gpu_algo_net_config_batch_varient_frequency(
#         'gtx2080ti', 'winograd_nonfused', net, winograd_nonfused_batch_sizes_list[i_net], gtx2080ti_coreF, gtx2080ti_memF[1])
#     draw_power_perf_fix_gpu_algo_net_config_batch_varient_frequency(
#         'gtx2080ti', 'winograd_nonfused', net, winograd_nonfused_batch_sizes_list[i_net], gtx2080ti_coreF, gtx2080ti_memF[1])

# for i_net, net in enumerate(nets):
#     draw_power_energy_fix_gpu_algo_net_config_batch_varient_frequency(
#         'gtx2080ti', 'fft_tile', net, fft_tile_batch_sizes_list[i_net], gtx2080ti_coreF, gtx2080ti_memF[1])
#     draw_power_perf_fix_gpu_algo_net_config_batch_varient_frequency(
#         'gtx2080ti', 'fft_tile', net, fft_tile_batch_sizes_list[i_net], gtx2080ti_coreF, gtx2080ti_memF[1])


















def draw_power_energy_fix_gpu_net_batch_config_algos_varient_frequency(
    gpu, algos, net, batch_size, gpu_coreF, gpu_memF, save=False):
    
    fig = plt.figure(figsize=(fig_x, fig_y))
    ax1 = fig.add_subplot(1, 1, 1)
    #==============
    ax2 = ax1.twinx()
    bar_i_width = 0

    bar_n = len(batch_sizes)
    bar_width = bar_total_width / bar_n
    np.array(gpu_coreF)
    # bar_x = np.array(gpu_coreF, np.dtype('int32')) - (bar_total_width - bar_width) / 2

    line_x = np.array([100, 200, 300, 400, 500, 600, 700], np.dtype('int32'))
    bar_x = line_x - (bar_total_width - bar_width) / 2

    #==============
    #==============
    max_power = 0
    max_energy = 0
    for i_algo, algo in enumerate(algos):
        powers_respect_coreF = get_power_respect(gpu, algo, 'caffe', net, batch_size, gpu_coreF, gpu_memF)
        power_label_name = '{2} {0}-b{1} Power'.format(net, batch_size, algo)
        max_power = max(powers_respect_coreF) if max_power < max(powers_respect_coreF) else max_power
        ax1.bar(bar_x + bar_i_width * bar_width, powers_respect_coreF, color=bar_colors[i_algo],
            width=bar_width, label=power_label_name)
        bar_i_width += 1

    for i_algo, algo in enumerate(algos):
        energy_respect_coreF = get_energy_respect(gpu, algo, 'caffe', net, batch_size, gpu_coreF, gpu_memF)       
        energy_label_name = '{2} {0}-b{1} Energy'.format(net, batch_size, algo)
        max_energy = max(energy_respect_coreF) if max_energy < max(energy_respect_coreF) else max_energy
        ax2.plot(line_x, energy_respect_coreF, color=plot_colors[i_algo], marker=markers[i_algo],
            linewidth=2, label=energy_label_name)
    
    # y1_intervals = int(max_power/12)
    # y1_loc = plticker.MultipleLocator(base=y1_intervals)
    # ax1.yaxis.set_major_locator(y1_loc)


    box = ax1.get_position()
    ax1.set_position([box.x0, box.y0, box.width, box.height * 0.8])
    ax1.legend(loc='upper left', bbox_to_anchor=(0., 0.98, 0.8, 0.1), ncol=4, mode="expand", borderaxespad=0., prop=font_legend)

    ax2.legend(loc='upper left', bbox_to_anchor=(0., 1.05, 0.8, 0.1), ncol=4, mode="expand", borderaxespad=0., prop=font_legend)
    # ax2.set_ylim([0, max_energy *1.1])
    ax2.set_ylabel("%s (J/image)" % 'energy', size=size_labels)

    ax1.set_yticks(np.linspace(0, max_power * y_max_ratio, num=y_axis_interval))
    ax2.set_yticks(np.linspace(0, max_energy * y_max_ratio, num=y_axis_interval))
    ax1.set_yticklabels(np.linspace(0, max_power * y_max_ratio, num=y_axis_interval).astype(np.int32), fontsize=size_ticks)
    ax2.set_yticklabels(np.linspace(0, max_energy * y_max_ratio, num=y_axis_interval).astype(np.int32), fontsize=size_ticks)
    ax1.grid(True, linestyle="--", color="0.5", linewidth=1)

    ax1.set_xlabel("%s (Hz)" % 'coreF', size=size_labels)
    ax1.set_ylabel("%s (J)" % 'power', size=size_labels)
    ax1.set_xticks(line_x)
    ax1.set_xticklabels([int(coreF) for coreF in gpu_coreF], size=size_ticks)
    # plt.xticks([int(coreF) for coreF in gpu_coreF], size=ticks_size)
    # plt.title("config [{0}, {1}], varient {2} ".format('net', 'batch', 'coreF'))

    if save == True:
        save_file_name = 'power&energy_ex3_{0}_{1}_b{2}'.format(gpu, net, batch_size)
        plt.savefig(os.path.join(OUTPUT_PATH, '%s.pdf' % save_file_name), bbox_inches='tight')
        plt.savefig(os.path.join(OUTPUT_PATH, '%s.png' % save_file_name), bbox_inches='tight')
    else:
        plt.show()

def draw_power_perf_fix_gpu_net_batch_config_algos_varient_frequency(
    gpu, algos, net, batch_size, gpu_coreF, gpu_memF, save=False):

    fig = plt.figure(figsize=(fig_x, fig_y))
    ax1 = fig.add_subplot(1, 1, 1)
    #==============
    ax2 = ax1.twinx()
    bar_i_width = 0

    bar_n = len(batch_sizes)
    bar_width = bar_total_width / bar_n
    np.array(gpu_coreF)
    # bar_x = np.array(gpu_coreF, np.dtype('int32')) - (bar_total_width - bar_width) / 2

    line_x = np.array([100, 200, 300, 400, 500, 600, 700], np.dtype('int32'))
    bar_x = line_x - (bar_total_width - bar_width) / 2

    #==============
    #==============
    max_power = 0
    max_image_perf = 0
    for i_algo, algo in enumerate(algos):
        powers_respect_coreF = get_power_respect(gpu, algo, 'caffe', net, batch_size, gpu_coreF, gpu_memF)
        power_label_name = '{2} {0}-b{1} Power'.format(net, batch_size, algo)
        max_power = max(powers_respect_coreF) if max_power < max(powers_respect_coreF) else max_power
        ax1.bar(bar_x + bar_i_width * bar_width, powers_respect_coreF, color=bar_colors[i_algo],
            width=bar_width, label=power_label_name)
        bar_i_width += 1

    for i_algo, algo in enumerate(algos):
        image_perf_respect_coreF = get_image_perf_respect(gpu, algo, 'caffe', net, batch_size, gpu_coreF, gpu_memF)       
        max_image_perf = max(image_perf_respect_coreF) if max_image_perf < max(image_perf_respect_coreF) else max_image_perf

        energy_label_name = '{2} {0}-b{1} image_perf'.format(net, batch_size, algo)
        ax2.plot(line_x, image_perf_respect_coreF, color=plot_colors[i_algo], marker=markers[i_algo],
            linewidth=2, label=energy_label_name)
    
    # y1_intervals = int(max_power/12)
    # y1_loc = plticker.MultipleLocator(base=y1_intervals)
    # ax1.yaxis.set_major_locator(y1_loc)


    box = ax1.get_position()
    ax1.set_position([box.x0, box.y0, box.width, box.height * 0.8])
    ax1.legend(loc='upper left', bbox_to_anchor=(0., 0.98, 0.8, 0.1), ncol=4, mode="expand", borderaxespad=0., prop=font_legend)

    ax2.legend(loc='upper left', bbox_to_anchor=(0., 1.05, 0.8, 0.1), ncol=4, mode="expand", borderaxespad=0., prop=font_legend)
    # ax2.set_ylim([0, max_image_perf*1.1])
    ax2.set_ylabel("%s (image/s)" % 'perf', size=size_labels)

    ax1.set_yticks(np.linspace(0, max_power * y_max_ratio, num=y_axis_interval))
    ax2.set_yticks(np.linspace(0, max_image_perf * y_max_ratio, num=y_axis_interval))
    ax1.set_yticklabels(np.linspace(0, max_power * y_max_ratio, num=y_axis_interval).astype(np.int32), fontsize=size_ticks)
    ax2.set_yticklabels(np.linspace(0, max_image_perf * y_max_ratio, num=y_axis_interval).astype(np.int32), fontsize=size_ticks)
    ax1.grid(True, linestyle="--", color="0.5", linewidth=1)

    ax1.set_xlabel("%s (Hz)" % 'coreF', size=size_labels)
    ax1.set_ylabel("%s (J)" % 'power', size=size_labels)
    ax1.set_xticks(line_x)
    ax1.set_xticklabels([int(coreF) for coreF in gpu_coreF], size=size_ticks)
    # plt.xticks([int(coreF) for coreF in gpu_coreF], size=ticks_size)
    # plt.title("config [{0}, {1}], varient {2} ".format('net', 'batch', 'coreF'))

    if save == True:
        save_file_name = 'power&perf_ex3_{0}_{1}_b{2}'.format(gpu, net, batch_size)
        plt.savefig(os.path.join(OUTPUT_PATH, '%s.pdf' % save_file_name), bbox_inches='tight')
        plt.savefig(os.path.join(OUTPUT_PATH, '%s.png' % save_file_name), bbox_inches='tight')
    else:
        plt.show()



# algos = ['ipc_gemm', 'fft_tile', 'winograd_nonfused']
# algos_nets = ['alexnet', 'alexnet', 'googlenet', 'googlenet']
# algos_batch_sizes = ['128', '256', '16', '32']
# for index, (net, batch_size) in enumerate(zip(algos_nets, algos_batch_sizes)):
#     draw_power_energy_fix_gpu_net_batch_config_algos_varient_frequency(
#         'p100', algos, net, batch_size, p100_coreF, p100_memF[0])
#     draw_power_perf_fix_gpu_net_batch_config_algos_varient_frequency(
#         'p100', algos, net, batch_size, p100_coreF, p100_memF[0])













# fig = plt.figure()
# y1_intervals = float(10000)
# y1_loc = plticker.MultipleLocator(base=y1_intervals)
# ax1 = fig.add_subplot(1, 1, 1)
# ax1.yaxis.set_major_locator(y1_loc)
# #==============
# ax2 = ax1.twinx()
# bar_i_width = 0
# bar_total_width = 130
# bar_n = len(nets) * len(batch_sizes)
# bar_width = bar_total_width / bar_n
# np.array(p100_coreF)
# bar_x = np.array(p100_coreF, np.dtype('int32')) - (bar_total_width - bar_width) / 2
# #==============
# for i_net, net in enumerate(nets):
#     for i_batch, batch in enumerate(batch_sizes_list[i_net]):
#         powers_respect_coreF = get_power_respect('p100', 'caffe', net, batch, p100_coreF, '715')
#         power_label_name = '{0}-b{1} Power'.format(net, batch)
#         bar_i_width += 1
#         ax1.bar(bar_x + bar_i_width * bar_width, powers_respect_coreF, color=bar_colors[i_batch],
#             hatch=hatches[i_net], width=bar_width, label=power_label_name)

# for i_net, net in enumerate(nets):
#     for i_batch, batch in enumerate(batch_sizes_list[i_net]):
#         energy_respect_coreF = get_energy_respect('p100', 'caffe', net, batch, p100_coreF, '715')       
#         energy_label_name = '{0}-b{1} Energy'.format(net, batch)
#         ax2.plot(np.array(p100_coreF, np.dtype('float')), energy_respect_coreF, color=plot_colors[i_batch], marker=markers[i_net],
#             linestyle=linestyles[i_net], linewidth=2, label=energy_label_name)

# ax1.grid(True, linestyle="--", color="0.5", linewidth=1)

# box = ax1.get_position()
# ax1.set_position([box.x0, box.y0, box.width, box.height * 0.8])
# ax1.legend(loc='upper left', bbox_to_anchor=(0., 1.1, 0.8, 0.1), ncol=5, mode="expand", borderaxespad=0.)

# ax2.legend(loc='upper left', bbox_to_anchor=(0., 1.25, 0.8, 0.1), ncol=5, mode="expand", borderaxespad=0.)
# ax2.set_ylim([0, 1000])
# ax2.set_ylabel("%s (image/s)" % 'perf')

# ax1.set_xlabel("%s (Hz)" % 'coreF')
# ax1.set_ylabel("%s (J)" % 'power')
# plt.xticks([int(coreF) for coreF in p100_coreF])
# # plt.title("config [{0}, {1}], varient {2} ".format('net', 'batch', 'coreF'))
# plt.show()




# fig = plt.figure()
# intervals = float(50)
# loc = plticker.MultipleLocator(base=intervals)
# ax = fig.add_subplot(1, 1, 1)
# ax.yaxis.set_major_locator(loc)
# for i_net, net in enumerate(nets):
#     print("=========***************============*************=============")
#     print("=========change net to  =%s ================================" % net)
#     print("=========***************============*************=============")
#     for i_batch, batch in enumerate(batch_sizes_list[i_net]):
#         image_perf_respect_coreF = get_image_perf_respect('p100', 'caffe', net, batch, p100_coreF, '715')
#         print("=========batch change , print perf====%s ========================" % batch)
#         print(image_perf_respect_coreF)
#         labelname = '{0}-b{1}'.format(net, batch)
#         ax.plot(p100_coreF, image_perf_respect_coreF, color=colors[i_batch], marker=markers[i_net],
#             linestyle=linestyles[i_net], linewidth=2, label=labelname)
# ax.grid(True, linestyle="--", color="0.5", linewidth=1)

# box = ax.get_position()
# ax.set_position([box.x0, box.y0, box.width, box.height * 0.8])
# ax.legend(loc='upper left', bbox_to_anchor=(0., 1.02, 1., .102), ncol=5, mode="expand", borderaxespad=0.)
# plt.xlabel("%s (Hz)" % 'coreF')
# plt.ylabel("%s (image/s)" % 'perf')
# # plt.title("config [{0}, {1}], varient {2} ".format('net', 'batch', 'coreF'))
# plt.show()



# fig = plt.figure()
# intervals = float(50)
# loc = plticker.MultipleLocator(base=intervals)
# ax = fig.add_subplot(1, 1, 1)
# ax.yaxis.set_major_locator(loc)
# for i_net, net in enumerate(nets):
#     print("=========***************============*************=============")
#     print("=========change net to  =%s ================================" % net)
#     print("=========***************============*************=============")
#     for i_batch, batch in enumerate(batch_sizes_list[i_net]):
#         energy_respect_coreF = get_energy_respect('p100', 'caffe', net, batch, p100_coreF, '715')
#         print("=========batch change , print energy =====%s ========================" % batch)
#         print(energy_respect_coreF)
#         labelname = '{0}-b{1}'.format(net, batch)
#         ax.plot(p100_coreF, energy_respect_coreF, color=colors[i_batch], marker=markers[i_batch],
#             linestyle=linestyles[i_net], linewidth=2, label=labelname)
# ax.grid(True, linestyle="--", color="0.5", linewidth=1)

# box = ax.get_position()
# ax.set_position([box.x0, box.y0, box.width, box.height * 0.8])
# ax.legend(loc='upper left', bbox_to_anchor=(0., 1.02, 1., .102), ncol=5, mode="expand", borderaxespad=0.)
# plt.xlabel("%s (Hz)" % 'coreF')
# plt.ylabel("%s (J/image)" % 'energy')
# # plt.title("config [{0}, {1}], varient {2} ".format('net', 'batch', 'coreF'))
# plt.show()



# fig = plt.figure()
# intervals = float(5)
# loc = plticker.MultipleLocator(base=intervals)
# ax = fig.add_subplot(1, 1, 1)
# ax.yaxis.set_major_locator(loc)
# for i_batch, batch in enumerate(alexnet_batch_sizes):
#     energy_respect_coreF = get_energy_respect('p100', 'caffe', 'alexnet', batch, p100_coreF, '715')
#     print("=========batch change , print energy =====%s ========================" % batch)
#     print(energy_respect_coreF)
#     labelname = '{0}-b{1}'.format('alexnet', batch)
#     ax.plot(p100_coreF, energy_respect_coreF, color=colors[i_batch], marker=markers[3],
#             linestyle=linestyles[0], linewidth=2, label=labelname)
# ax.grid(True, linestyle="--", color="0.5", linewidth=1)

# box = ax.get_position()
# ax.set_position([box.x0, box.y0, box.width, box.height * 0.8])
# ax.legend(loc='upper left', bbox_to_anchor=(0., 1.02, 1., .102), ncol=5, mode="expand", borderaxespad=0.)
# plt.xlabel("%s (Hz)" % 'coreF')
# plt.ylabel("%s (J/image)" % 'energy')
# #plt.title("config [{0}, {1}], varient {2} ".format('net', 'batch', 'coreF'))
# plt.show()



