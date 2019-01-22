python dvfs_benchmark.py --gpu-setting gtx2080ti --dl-setting dl_settings --algo auto --datapath C:/fake_train
python dvfs_benchmark.py --gpu-setting gtx2080ti --dl-setting dl_settings --algo ipc_gemm --datapath C:/fake_train
python dvfs_benchmark.py --gpu-setting gtx2080ti --dl-setting dl_settings --algo fft_tile --datapath C:/fake_train
python dvfs_benchmark.py --gpu-setting gtx2080ti --dl-setting dl_settings --algo winograd_nonfused --datapath C:/fake_train

python dvfs_benchmark.py --gpu-setting gtx1080ti --dl-setting dl_settings --algo auto --datapath C:/fake_train
python dvfs_benchmark.py --gpu-setting gtx1080ti --dl-setting dl_settings --algo ipc_gemm --datapath C:/fake_train
python dvfs_benchmark.py --gpu-setting gtx1080ti --dl-setting dl_settings --algo fft_tile --datapath C:/fake_train
python dvfs_benchmark.py --gpu-setting gtx1080ti --dl-setting dl_settings --algo winograd_nonfused --datapath C:/fake_train
