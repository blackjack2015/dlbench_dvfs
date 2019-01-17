# dlbench_dvfs
Deep Learning with GPU DVFS

gpuname.cfg 
    iters = 200
    secs = 40
    cuda_device_id = 1
    nvIns_device_id = 0
    rest_time = 2
    power_sample_interval = 50

nvidia-smi -i 0 -q -d SUPPORTED_FREQUENCY 

dl_setting.cfg
