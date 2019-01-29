rem python to_txt.py -d p100_auto_out
rem python to_txt.py -d p100_ipc_gemm_out
rem python to_txt.py -d p100_fft_tile_out
rem python to_txt.py -d p100_winograd_nonfused_out

python to_txt.py -d v100_auto_out
python to_txt.py -d v100_ipc_gemm_out
python to_txt.py -d v100_fft_tile_out
python to_txt.py -d v100_winograd_nonfused_out

