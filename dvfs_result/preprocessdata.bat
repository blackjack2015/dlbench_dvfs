python get_caffe_performance.py -d p100_auto
python get_power.py -d p100_auto
python get_data.py -d p100_auto

python get_caffe_performance.py -d p100_ipc_gemm
python get_power.py -d p100_ipc_gemm
python get_data.py -d p100_ipc_gemm

python get_caffe_performance.py -d p100_fft_tile
python get_power.py -d p100_fft_tile
python get_data.py -d p100_fft_tile

python get_caffe_performance.py -d p100_winograd_nonfused
python get_power.py -d p100_winograd_nonfused
python get_data.py -d p100_winograd_nonfused

python get_caffe_performance.py -d v100_auto
python get_power.py -d v100_auto
python get_data.py -d v100_auto

python get_caffe_performance.py -d v100_ipc_gemm
python get_power.py -d v100_ipc_gemm
python get_data.py -d v100_ipc_gemm

python get_caffe_performance.py -d v100_fft_tile
python get_power.py -d v100_fft_tile
python get_data.py -d v100_fft_tile

python get_caffe_performance.py -d v100_winograd_nonfused
python get_power.py -d v100_winograd_nonfused
python get_data.py -d v100_winograd_nonfused

