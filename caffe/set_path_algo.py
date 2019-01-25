import glob, os, re
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--algo', type=str, help='setting cudnn conv algorithm.', default='auto')
parser.add_argument('--datapath', type=str, help='setting datapath for caffe prototxt.', default='/home/hpcl/data/caffe/fake_image_net.lmdb')

network_filelist = glob.glob(r'networks/*.prototxt')
network_filelist.sort()

opt = parser.parse_args()
datapath = opt.datapath
algo = opt.algo
#algo = "ipc_gemm" # auto, find, ipc_gemm, fft_tile, winograd_nonfused, winograd

def set_network(filepath, datapath, algo):

    replacement_list = {
        #'$TRAIN_PATH': ('%s' % datapath),
        #'$TEST_PATH': ('%s' % datapath),
        'type: \"Convolution\"\n': ('type: \"Convolution\"\n  cudnn_algo: \"%s\"\n' % algo),
    }

    proto = ''
    tfile = open(filepath, "r")
    proto = tfile.read()
    tfile.close()

    for r in replacement_list:
        proto = proto.replace(r, replacement_list[r])
    
    tmpfile, number = re.subn('networks', 'tmp', filepath)
    tfile = open(tmpfile, "w")
    tfile.write(proto)
    tfile.close()

if not os.path.exists('tmp'):
    os.makedirs('tmp')

for fp in network_filelist:
    set_network(fp, datapath, algo)
