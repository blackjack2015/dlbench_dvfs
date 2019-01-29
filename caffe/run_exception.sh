datapath=/home/hpcl/data/caffe/fake_image_net.lmdb


# python dvfs_benchmark.py --gpu-setting p100_f810 --dl-setting dl_setting_p100_f810_winograd --algo winograd_nonfused --datapath $datapath


# python dvfs_benchmark.py --gpu-setting v100_f510 --dl-setting dl_setting_v100_f510_ipc_gemm --algo ipc_gemm --datapath $datapath
# python dvfs_benchmark.py --gpu-setting v100_f652 --dl-setting dl_setting_v100_f652_ipc_gemm --algo ipc_gemm --datapath $datapath
# python dvfs_benchmark.py --gpu-setting v100_f802 --dl-setting dl_setting_v100_f802_ipc_gemm --algo ipc_gemm --datapath $datapath
python dvfs_benchmark.py --gpu-setting v100_f945 --dl-setting dl_setting_v100_f945_ipc_gemm --algo ipc_gemm --datapath $datapath
# python dvfs_benchmark.py --gpu-setting v100_f1087 --dl-setting dl_setting_v100_f1087_ipc_gemm --algo ipc_gemm --datapath $datapath
# python dvfs_benchmark.py --gpu-setting v100_f1237 --dl-setting dl_setting_v100_f1237_ipc_gemm --algo ipc_gemm --datapath $datapath
python dvfs_benchmark.py --gpu-setting v100_f1380 --dl-setting dl_setting_v100_f1380_ipc_gemm --algo ipc_gemm --datapath $datapath

# python dvfs_benchmark.py --gpu-setting v100_f510 --dl-setting dl_setting_v100_f510_winograd --algo winograd --datapath $datapath
# python dvfs_benchmark.py --gpu-setting v100_f652 --dl-setting dl_setting_v100_f652_winograd --algo winograd --datapath $datapath
# python dvfs_benchmark.py --gpu-setting v100_f802 --dl-setting dl_setting_v100_f802_winograd --algo winograd --datapath $datapath
# python dvfs_benchmark.py --gpu-setting v100_f945 --dl-setting dl_setting_v100_f945_winograd --algo winograd --datapath $datapath
# python dvfs_benchmark.py --gpu-setting v100_f1087 --dl-setting dl_setting_v100_f1087_winograd --algo winograd --datapath $datapath
# python dvfs_benchmark.py --gpu-setting v100_f1237 --dl-setting dl_setting_v100_f1237_winograd --algo winograd --datapath $datapath
python dvfs_benchmark.py --gpu-setting v100_f1380 --dl-setting dl_setting_v100_f1380_winograd --algo winograd --datapath $datapath

# python dvfs_benchmark.py --gpu-setting v100_f510 --dl-setting dl_setting_v100_f510_fft_tile --algo fft_tile --datapath $datapath
# python dvfs_benchmark.py --gpu-setting v100_f652 --dl-setting dl_setting_v100_f652_fft_tile --algo fft_tile --datapath $datapath
# python dvfs_benchmark.py --gpu-setting v100_f802 --dl-setting dl_setting_v100_f802_fft_tile --algo fft_tile --datapath $datapath
python dvfs_benchmark.py --gpu-setting v100_f945 --dl-setting dl_setting_v100_f945_fft_tile --algo fft_tile --datapath $datapath
# python dvfs_benchmark.py --gpu-setting v100_f1087 --dl-setting dl_setting_v100_f1087_fft_tile --algo fft_tile --datapath $datapath
# python dvfs_benchmark.py --gpu-setting v100_f1237 --dl-setting dl_setting_v100_f1237_fft_tile --algo fft_tile --datapath $datapath
# python dvfs_benchmark.py --gpu-setting v100_f1380 --dl-setting dl_setting_v100_f1380_fft_tile --algo fft_tile --datapath $datapath
