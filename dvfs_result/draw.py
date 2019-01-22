import os, sys
import argparse
import re
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='GPU measure')
parser.add_argument('-d', '--data-dir', default='.', type=str,
                    help='sssss')
args = parser.parse_args()
data_dir = args.data_dir

gpu = ['p100', 'v100']
nets = ['alexnet', 'resnet', 'vggnet', 'googlenet']
p100_coreF = ['544', '683', '810', '936', '1063', '1202', '1328']
v100_coreF = ['607', '810', '1012', '1200', '1327', '1432', '1530']
p100_memF = ['715']
v100_memF = ['877']
batch_sizes = ['16', '32', '64', '128']
alexnet_batch_sizes = ['128', '256', '512', '1024']

variens_list = ['gpu', 'framework', 'net', 'batch_size', 'coreF', 'memF']

colors = ['b', 'g', 'r', 'k', 'c', 'm', 'y']
markers = ['o', 'v', 's', '*']

def get_batch_time(gpu, framework, net, batch_size, coreF, memF):
    gpu_dir = os.path.join(data_dir, gpu + '_out')
    data_name = framework + '_' + net + '_' + batch_size + '_' + coreF + '_' + memF + '_perf.log'
    data_path = os.path.join(gpu_dir, data_name)
    data = open(data_path, 'r').readline().split()[0]
    return float(data)

def get_image_perf(gpu, framework, net, batch_size, coreF, memF):
    gpu_dir = os.path.join(data_dir, gpu + '_out')
    data_name = framework + '_' + net + '_' + batch_size + '_' + coreF + '_' + memF + '_perf.log'
    data_path = os.path.join(gpu_dir, data_name)
    data = open(data_path, 'r').readline().split()[1]
    return float(data)

def get_power(gpu, framework, net, batch_size, coreF, memF):
    gpu_dir = os.path.join(data_dir, gpu + '_out')
    data_name = framework + '_' + net + '_' + batch_size + '_' + coreF + '_' + memF + '_power.log'
    data_path = os.path.join(gpu_dir, data_name)
    data = open(data_path, 'r').readline().split()[0]
    return float(data)

def get_energy(gpu, framework, net, batch_size, coreF, memF):
    power = get_power(gpu, framework, net, batch_size, coreF, memF)
    image_perf = get_image_perf(gpu, framework, net, batch_size, coreF, memF)
    return (power / image_perf)

def get_power_respect(gpu=None, framework=None, net=None, batch_size=None, coreF=None, memF=None):
    '''
        just one varient can be list
    '''
    powers = []
    varients = [gpu, framework, net, batch_size, coreF, memF]
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


def get_batch_time_respect(gpu=None, framework=None, net=None, batch_size=None, coreF=None, memF=None):
    '''
        just one varient can be list
    '''
    powers = []
    varients = [gpu, framework, net, batch_size, coreF, memF]
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



def get_image_perf_respect(gpu=None, framework=None, net=None, batch_size=None, coreF=None, memF=None):
    '''
        just one varient can be list
    '''
    powers = []
    varients = [gpu, framework, net, batch_size, coreF, memF]
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

def get_energy_respect(gpu=None, framework=None, net=None, batch_size=None, coreF=None, memF=None):
    powers = get_power_respect(gpu, framework, net, batch_size, coreF, memF)
    image_perfs = get_image_perf_respect(gpu, framework, net, batch_size, coreF, memF)
    energys = []
    for i in range(len(powers)):
        energys.append(powers[i] / image_perfs[i])
    return energys


test_get_power = get_power('p100', 'caffe', 'alexnet', '1024', '1063', '715')
test_get_powers = get_power_respect('p100', 'caffe', 'alexnet', '1024', p100_coreF, '715')
test_get_powers_alex = get_power_respect('p100', 'caffe', 'alexnet', alexnet_batch_sizes, '1063', '715')
test_get_powers_resnet = get_power_respect('p100', 'caffe', 'resnet', batch_sizes, '1063', '715')
test_get_powers_google = get_power_respect('p100', 'caffe', 'googlenet', batch_sizes, '1063', '715')
test_get_image_perf_respect_batch = get_image_perf_respect('p100', 'caffe', 'googlenet', batch_sizes, '1063', '715')
test_get_energy_respect_batch = get_energy_respect('p100', 'caffe', 'googlenet', batch_sizes, '1063', '715')

# print(test_get_power)
# print(test_get_powers)
# print(test_get_powers_alex)
# print(test_get_powers_resnet)
# print(test_get_powers_google)
# print(test_get_image_perf_respect_batch)
# print(test_get_energy_respect_batch)
for coreF in p100_coreF:
    print("=========***************============*************=============")
    print("=========change coreF to  =%s ================================" % coreF)
    print("=========***************============*************=============")
    for net in ['vggnet', 'googlenet']:
        test_get_powers_respect_batch = get_power_respect('p100', 'caffe', net, batch_sizes, coreF, '715')
        test_get_image_perf_respect_batch = get_image_perf_respect('p100', 'caffe', net, batch_sizes, coreF, '715')
        test_get_energy_respect_batch = get_energy_respect('p100', 'caffe', net, batch_sizes, coreF, '715')
        print("=========batch change , print power, perf, energy =====%s ========================" % net)
        print(test_get_powers_respect_batch)
        print(test_get_image_perf_respect_batch)
        print(test_get_energy_respect_batch)


    test_get_powers_respect_batch = get_power_respect('p100', 'caffe', 'alexnet', alexnet_batch_sizes, coreF, '715')
    test_get_image_perf_respect_batch = get_image_perf_respect('p100', 'caffe', 'alexnet', alexnet_batch_sizes, coreF, '715')
    test_get_energy_respect_batch = get_energy_respect('p100', 'caffe', 'alexnet', alexnet_batch_sizes, coreF, '715')
    print("=========batch change , print power, perf, energy =====%s ========================" % 'alexnet')
    print(test_get_powers_respect_batch)
    print(test_get_image_perf_respect_batch)
    print(test_get_energy_respect_batch)

plt.figure()
for i_net, net in enumerate(['vggnet', 'googlenet']):
    print("=========***************============*************=============")
    print("=========change net to  =%s ================================" % net)
    print("=========***************============*************=============")
    for i_batch, batch in enumerate(batch_sizes):
        powers_respect_coreF = get_power_respect('p100', 'caffe', net, batch, p100_coreF, '715')
        print("=========coreF change , print power====%s ========================" % batch)
        print(powers_respect_coreF)
        labelname = '{0}-b{1}'.format(net, batch)
        plt.plot(p100_coreF, powers_respect_coreF, color=colors[i_batch], marker=markers[i_net],
            linewidth=1.0, label=labelname)

plt.xlabel("%s (Hz)" % 'coreF')
plt.ylabel("%s (J)" % 'power')
plt.title("config [{0}, {1}], varient {2} ".format('net', 'batch', 'coreF'))
plt.legend(loc='upper left')
plt.show()


plt.figure()
for i_net, net in enumerate(['vggnet', 'googlenet']):
    print("=========***************============*************=============")
    print("=========change net to  =%s ================================" % net)
    print("=========***************============*************=============")
    for i_batch, batch in enumerate(batch_sizes):
        image_perf_respect_coreF = get_image_perf_respect('p100', 'caffe', net, batch, p100_coreF, '715')
        print("=========batch change , print perf====%s ========================" % batch)
        print(image_perf_respect_coreF)
        labelname = '{0}-b{1}'.format(net, batch)
        plt.plot(p100_coreF, image_perf_respect_coreF, color=colors[i_batch], marker=markers[i_net],
            linewidth=1.0, label=labelname)

plt.xlabel("%s (Hz)" % 'coreF')
plt.ylabel("%s (image/s)" % 'perf')
plt.title("config [{0}, {1}], varient {2} ".format('net', 'batch', 'coreF'))
plt.legend(loc='upper left')
plt.show()

plt.figure()
for i_net, net in enumerate(['vggnet', 'googlenet']):
    print("=========***************============*************=============")
    print("=========change net to  =%s ================================" % net)
    print("=========***************============*************=============")
    for i_batch, batch in enumerate(batch_sizes):
        energy_respect_coreF = get_energy_respect('p100', 'caffe', net, batch, p100_coreF, '715')
        print("=========batch change , print energy =====%s ========================" % batch)
        print(energy_respect_coreF)
        labelname = '{0}-b{1}'.format(net, batch)
        plt.plot(p100_coreF, energy_respect_coreF, color=colors[i_batch], marker=markers[i_net],
            linewidth=1.0, label=labelname)

plt.xlabel("%s (Hz)" % 'coreF')
plt.ylabel("%s (J/image)" % 'energy')
plt.title("config [{0}, {1}], varient {2} ".format('net', 'batch', 'coreF'))
plt.legend(loc='upper left')
plt.show()
