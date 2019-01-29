import os, sys
import argparse
import re
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
import numpy as np
import draw as data

parser = argparse.ArgumentParser(description='GPU measure')
parser.add_argument('-d', '--data-dir', default='.', type=str,
                    help='sssss')
args = parser.parse_args()
data_dir = args.data_dir
OUTPUT_PATH = 'pictures_out'

gpus = ['p100', 'v100', 'gtx2080ti']
algos = ['ipc_gemm', 'winograd', 'fft_tile']
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



# ====================***************************************************************===========================
#=====================================
gtx_ipc_gemm_alexnet_batch_sizes = ['64', '128', '256', '512']
gtx_ipc_gemm_resnet_batch_sizes = ['8', '16']
gtx_ipc_gemm_vggnet_batch_sizes = ['16', '32', '64']
gtx_ipc_gemm_googlenet_batch_sizes = ['16', '32', '64', '128']
gtx_ipc_gemm_batch_sizes_list = [gtx_ipc_gemm_alexnet_batch_sizes, gtx_ipc_gemm_resnet_batch_sizes, 
                        gtx_ipc_gemm_vggnet_batch_sizes, gtx_ipc_gemm_googlenet_batch_sizes]
gtx_ipc_gemm_nets = ['alexnet', 'resnet', 'vggnet', 'googlenet']
#=======================================
gtx_winograd_nonfused_alexnet_batch_sizes = ['64', '128', '256']
gtx_winograd_nonfused_resnet_batch_sizes = ['8', '16']
gtx_winograd_nonfused_vggnet_batch_sizes = [ ]
gtx_winograd_nonfused_googlenet_batch_sizes = ['16', '32', '64']
gtx_winograd_nonfused_batch_sizes_list = [gtx_winograd_nonfused_alexnet_batch_sizes, gtx_winograd_nonfused_resnet_batch_sizes, 
                        gtx_winograd_nonfused_vggnet_batch_sizes, gtx_winograd_nonfused_googlenet_batch_sizes]
gtx_winograd_nets = ['alexnet', 'resnet', 'googlenet']
#========================================
gtx_fft_tile_alexnet_batch_sizes = ['64']
gtx_fft_tile_resnet_batch_sizes = [  ]
gtx_fft_tile_vggnet_batch_sizes = [  ]
gtx_fft_tile_googlenet_batch_sizes = ['16']
gtx_fft_tile_batch_sizes_list = [gtx_fft_tile_alexnet_batch_sizes, gtx_fft_tile_googlenet_batch_sizes]
gtx_fft_tile_nets = ['alexnet', 'googlenet']
#=========================================

gtx_algo_nets = [gtx_ipc_gemm_nets, gtx_winograd_nets, gtx_fft_tile_nets]
gtx_algo_batch_sizes_list = [gtx_ipc_gemm_batch_sizes_list, gtx_winograd_nonfused_batch_sizes_list, gtx_fft_tile_batch_sizes_list]






# ====================***************************************************************===========================



variens_list = ['gpu', 'algo', 'framework', 'net', 'batch_size', 'coreF', 'memF']

algo_index = {
    'ipc_gemm': 0,
    'winograd': 1,
    'fft_tile': 2
}


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
'size' : 15,
}
font_ticks = {'family' : 'Times New Roman',
'weight' : 'normal',
'size' : 15,
}
font_labels = {'family' : 'Times New Roman',
'weight' : 'normal',
'size' : 26,
}

fig_x = 12
fig_y = 8
y_axis_interval = 10
y_max_ratio = 1.1

#======================================
# bar_colors = ['lightcoral', 'burlywood', 'y', 'yellowgreen',
#             'lightgreen', 'lightseagreen', 'lightskyblue', 'mediumpurple']
bar_colors = ['#0A64A4', '#24577B', '#03406A', '#3E94D1']
hatches = ['-', '+', 'x', '\\', '|', '/', 'O', '.']


def get_gpu_algo_geo_mean_power(gpu, algo):
    i_algo = algo_index[algo]
    i_gpu = gpu_index[gpu]

    local_algo_batch_sizes_list = algo_batch_sizes_list
    local_algo_nets = algo_nets

    if gpu == 'gtx2080ti':
        local_algo_batch_sizes_list = gtx_algo_batch_sizes_list
        local_algo_nets = gtx_algo_nets
        results = np.array([1, 1, 1, 1, 1, 1], np.dtype('float'))
    else:
        results = np.array([1, 1, 1, 1, 1, 1, 1], np.dtype('float'))

    n = 0
    for i_net, net in enumerate(local_algo_nets[i_algo]):
        n += len(local_algo_batch_sizes_list[i_algo][i_net])

    for i_net, net in enumerate(local_algo_nets[i_algo]):
        for i_batch, batch in enumerate(local_algo_batch_sizes_list[i_algo][i_net]):
            powers_respect_coreF = np.array(data.get_power_respect(gpu, algo, 'caffe', net, batch, gpu_coreFs[i_gpu], gpu_memFs[i_gpu][0]), np.dtype('float'))
            powers_sqrt = np.power(powers_respect_coreF, 1/n)
            results = results * powers_sqrt

    return results

def get_gpu_algo_geo_mean_perf(gpu, algo):
    i_algo = algo_index[algo]
    i_gpu = gpu_index[gpu]

    local_algo_batch_sizes_list = algo_batch_sizes_list
    local_algo_nets = algo_nets

    if gpu == 'gtx2080ti':
        local_algo_batch_sizes_list = gtx_algo_batch_sizes_list
        local_algo_nets = gtx_algo_nets
        results = np.array([1, 1, 1, 1, 1, 1], np.dtype('float'))
    else:
        results = np.array([1, 1, 1, 1, 1, 1, 1], np.dtype('float'))

    n = 0
    for i_net, net in enumerate(local_algo_nets[i_algo]):
        n += len(local_algo_batch_sizes_list[i_algo][i_net])

    for i_net, net in enumerate(local_algo_nets[i_algo]):
        for i_batch, batch in enumerate(local_algo_batch_sizes_list[i_algo][i_net]):
            perfs_respect_coreF = np.array(data.get_image_perf_respect(gpu, algo, 'caffe', net, batch, gpu_coreFs[i_gpu], gpu_memFs[i_gpu][0]), np.dtype('float'))
            perfs_sqrt = np.power(perfs_respect_coreF, 1/n)
            results = results * perfs_sqrt

    return results

def get_gpu_algo_geo_mean_energy(gpu, algo):
    i_algo = algo_index[algo]
    i_gpu = gpu_index[gpu]

    local_algo_batch_sizes_list = algo_batch_sizes_list
    local_algo_nets = algo_nets

    if gpu == 'gtx2080ti':
        local_algo_batch_sizes_list = gtx_algo_batch_sizes_list
        local_algo_nets = gtx_algo_nets
        results = np.array([1, 1, 1, 1, 1, 1], np.dtype('float'))
    else:
        results = np.array([1, 1, 1, 1, 1, 1, 1], np.dtype('float'))

    n = 0
    for i_net, net in enumerate(local_algo_nets[i_algo]):
        n += len(local_algo_batch_sizes_list[i_algo][i_net])

    for i_net, net in enumerate(local_algo_nets[i_algo]):
        for i_batch, batch in enumerate(local_algo_batch_sizes_list[i_algo][i_net]):
            energys_respect_coreF = np.array(data.get_energy_respect(gpu, algo, 'caffe', net, batch, gpu_coreFs[i_gpu], gpu_memFs[i_gpu][0]), np.dtype('float'))
            energys_sqrt = np.power(energys_respect_coreF, 1/n)
            results = results * energys_sqrt

    return results




def draw_power_energy_fix_gpu_config_algos_varient_frequency(
    gpu, save=False):
    
    i_gpu = gpu_index[gpu]
    gpu_coreF = gpu_coreFs[i_gpu]

    fig = plt.figure(figsize=(fig_x, fig_y))
    ax1 = fig.add_subplot(1, 1, 1)
    #==============
    ax2 = ax1.twinx()
    bar_i_width = 0
    bar_total_width = 100
    bar_n = len(algos)

    bar_width = bar_total_width / bar_n
    np.array(gpu_coreF)
    bar_x = np.array(gpu_coreF, np.dtype('int32')) - (bar_total_width - bar_width) / 2
    #==============
    max_power = 0
    max_energy = 0

    for algo in algos:
        i_algo = algo_index[algo]

        powers_respect_coreF = get_gpu_algo_geo_mean_power(gpu, algo)
        power_label_name = '{0} Power'.format(algo)

        max_power = max(powers_respect_coreF) if max_power < max(powers_respect_coreF) else max_power
        bar_i_width += 1
        ax1.bar(bar_x + bar_i_width * bar_width, powers_respect_coreF, color=bar_colors[i_algo],
            width=bar_width, label=power_label_name)



    for algo in algos:
        i_algo = algo_index[algo] 
        energy_respect_coreF = get_gpu_algo_geo_mean_energy(gpu, algo)
        energy_label_name = '{0} Energy'.format(algo)
        max_energy = max(energy_respect_coreF) if max_energy < max(energy_respect_coreF) else max_energy
        ax2.plot(np.array(gpu_coreF, np.dtype('float')), energy_respect_coreF, color=plot_colors[i_algo], marker=markers[i_algo],
            linewidth=2, label=energy_label_name)
    
    # y1_intervals = int(max_power/12)
    # y1_loc = plticker.MultipleLocator(base=y1_intervals)
    # ax1.yaxis.set_major_locator(y1_loc)
    # ax1.grid(True, linestyle="--", color="0.5", linewidth=1)

    box = ax1.get_position()
    ax1.set_position([box.x0, box.y0, box.width, box.height * 0.8])
    ax1.legend(loc='upper left', bbox_to_anchor=(0., 0.98, 0.8, 0.1), ncol=4, mode="expand", borderaxespad=0., prop=font_legend)

    ax2.legend(loc='upper left', bbox_to_anchor=(0., 1.05, 0.8, 0.1), ncol=4, mode="expand", borderaxespad=0., prop=font_legend)

    ax1.set_yticks(np.linspace(0, max_power * y_max_ratio, num=y_axis_interval))
    ax2.set_yticks(np.linspace(0, max_energy * y_max_ratio, num=y_axis_interval))
    ax1.set_yticklabels(np.linspace(0, max_power * y_max_ratio, num=y_axis_interval).astype(np.int32), fontsize=size_ticks)
    ax2.set_yticklabels(np.linspace(0, max_energy * y_max_ratio, num=y_axis_interval).astype(np.int32), fontsize=size_ticks)
    ax1.grid(True, linestyle="--", color="0.5", linewidth=1)

    ax2.set_ylabel("%s (J/image)" % 'energy', size=size_labels)

    ax1.set_xlabel("%s (Hz)" % 'coreF', size=size_labels)
    ax1.set_ylabel("%s (J)" % 'power', size=size_labels)
    ax1.set_xticks([int(coreF) for coreF in gpu_coreF])
    ax1.set_xticklabels([int(coreF) for coreF in gpu_coreF], size=size_ticks)

    # plt.xticks([int(coreF) for coreF in gpu_coreF], size=ticks_size)
    # plt.title("config [{0}, {1}], varient {2} ".format('net', 'batch', 'coreF'))
    if save == True:
        save_file_name = 'power&energy_ex1_{0}_{1}_{2}'.format(gpu, algo, net)
        plt.savefig(os.path.join(OUTPUT_PATH, '%s.pdf' % save_file_name), bbox_inches='tight')
        plt.savefig(os.path.join(OUTPUT_PATH, '%s.png' % save_file_name), bbox_inches='tight')
    else:
        plt.show()



def draw_power_perf_fix_gpu_config_algos_varient_frequency(
    gpu, save=False):

    i_gpu = gpu_index[gpu]
    gpu_coreF = gpu_coreFs[i_gpu]

    fig = plt.figure(figsize=(fig_x, fig_y))
    ax1 = fig.add_subplot(1, 1, 1)
    #==============
    ax2 = ax1.twinx()
    bar_i_width = 0
    bar_total_width = 100
    bar_n = len(algos)

    bar_width = bar_total_width / bar_n
    np.array(gpu_coreF)
    bar_x = np.array(gpu_coreF, np.dtype('int32')) - (bar_total_width - bar_width) / 2
    #==============
    max_power = 0
    max_image_perf = 0

    for algo in algos:
        i_algo = algo_index[algo]

        powers_respect_coreF = get_gpu_algo_geo_mean_power(gpu, algo)
        power_label_name = '{0} Power'.format(algo)

        max_power = max(powers_respect_coreF) if max_power < max(powers_respect_coreF) else max_power
        bar_i_width += 1
        ax1.bar(bar_x + bar_i_width * bar_width, powers_respect_coreF, color=bar_colors[i_algo],
            width=bar_width, label=power_label_name)

    for algo in algos:
        i_algo = algo_index[algo]

        image_perf_respect_coreF = get_gpu_algo_geo_mean_perf(gpu, algo)
        max_image_perf = max(image_perf_respect_coreF) if max_image_perf < max(image_perf_respect_coreF) else max_image_perf

        energy_label_name = '{0} image_per'.format(algo)
        ax2.plot(np.array(gpu_coreF, np.dtype('float')), image_perf_respect_coreF, color=plot_colors[i_algo], marker=markers[i_algo],
            linewidth=2, label=energy_label_name)
    
    # y1_intervals = int(max_power/12)
    # y1_loc = plticker.MultipleLocator(base=y1_intervals)
    # ax1.yaxis.set_major_locator(y1_loc)
    # ax1.grid(True, linestyle="--", color="0.5", linewidth=1)

    box = ax1.get_position()
    ax1.set_position([box.x0, box.y0, box.width, box.height * 0.8])
    ax1.legend(loc='upper left', bbox_to_anchor=(0., 0.98, 0.8, 0.1), ncol=4, mode="expand", borderaxespad=0., prop=font_legend)
    ax2.legend(loc='upper left', bbox_to_anchor=(0., 1.05, 0.8, 0.1), ncol=4, mode="expand", borderaxespad=0., prop=font_legend)
    
    ax1.set_yticks(np.linspace(0, max_power * y_max_ratio, num=y_axis_interval))
    ax2.set_yticks(np.linspace(0, max_image_perf * y_max_ratio, num=y_axis_interval))
    ax1.set_yticklabels(np.linspace(0, max_power * y_max_ratio, num=y_axis_interval).astype(np.int32), fontsize=size_ticks)
    ax2.set_yticklabels(np.linspace(0, max_image_perf * y_max_ratio, num=y_axis_interval).astype(np.int32), fontsize=size_ticks)
    ax1.grid(True, linestyle="--", color="0.5", linewidth=1)

    ax2.set_ylabel("%s (image/s)" % 'perf', size=size_labels)
    ax1.set_xlabel("%s (Hz)" % 'coreF', size=size_labels)
    ax1.set_ylabel("%s (J)" % 'power', size=size_labels)
    ax1.set_xticks([int(coreF) for coreF in gpu_coreF])
    ax1.set_xticklabels([int(coreF) for coreF in gpu_coreF], size=size_ticks)
    # plt.xticks([int(coreF) for coreF in gpu_coreF])
    # plt.title("config [{0}, {1}], varient {2} ".format('net', 'batch', 'coreF'))
    if save == True:
        save_file_name = 'power&perf_ex1_{0}_{1}_{2}'.format(gpu, algo, net)
        plt.savefig(os.path.join(OUTPUT_PATH, '%s.pdf' % save_file_name), bbox_inches='tight')
        plt.savefig(os.path.join(OUTPUT_PATH, '%s.png' % save_file_name), bbox_inches='tight')
    else:
        plt.show()




# test = get_gpu_algo_geo_mean_power('v100', 'ipc_gemm')
# print(test)


for i_gpu, gpu in enumerate(gpus):

    draw_power_energy_fix_gpu_config_algos_varient_frequency(gpu)
    draw_power_perf_fix_gpu_config_algos_varient_frequency(gpu)



