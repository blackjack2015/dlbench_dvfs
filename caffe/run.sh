datapath=/home/hpcl/data/caffe/fake_image_net.lmdb

python dvfs_benchmark.py --gpu-setting v100 --dl-setting dl_settings --algo auto --datapath $datapath
python dvfs_benchmark.py --gpu-setting v100 --dl-setting dl_settings --algo ipc_gemm --datapath $datapath
python dvfs_benchmark.py --gpu-setting v100 --dl-setting dl_settings --algo fft_tile --datapath --datapath $datapath
python dvfs_benchmark.py --gpu-setting v100 --dl-setting dl_settings --algo winograd_nonfused --datapath $datapath

python dvfs_benchmark.py --gpu-setting p100 --dl-setting dl_settings --algo auto --datapath $datapath
python dvfs_benchmark.py --gpu-setting p100 --dl-setting dl_settings --algo ipc_gemm --datapath $datapath
python dvfs_benchmark.py --gpu-setting p100 --dl-setting dl_settings --algo fft_tile --datapath $datapath
python dvfs_benchmark.py --gpu-setting p100 --dl-setting dl_settings --algo winograd_nonfused --datapath $datapath
