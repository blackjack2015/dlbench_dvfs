import os,sys
import subprocess
import time
import re
import ConfigParser
import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dl-setting', type=str, help='benchmark setting of deep learning', default='dl_setting')
parser.add_argument('--gpu-setting', type=str, help='gpu card', default='gtx1080ti')
parser.add_argument('--algo', type=str, help='conv algo', default='auto') # auto, find, ipc_gemm, fft_tile, winograd_nonfused
parser.add_argument('--datapath', type=str, help='datapath for caffe', default='C:/fake_train') # auto, find, ipc_gemm, fft_tile, winograd_nonfused

opt = parser.parse_args()
print opt

# set path and conv algo for caffe prototxt
conv_algo = opt.algo
datapath = opt.datapath
os.system("python set_path_algo.py --algo %s --datapath %s" % (conv_algo, datapath))

if sys.platform.startswith("win"):
    # Don't display the Windows GPF dialog if the invoked program dies.
    # See comp.os.ms-windows.programmer.win32
    #  How to suppress crash notification dialog?, Jan 14,2004 -
    #     Raymond Chen's response [1]

    import ctypes
    SEM_NOGPFAULTERRORBOX = 0x0002 # From MSDN
    ctypes.windll.kernel32.SetErrorMode(SEM_NOGPFAULTERRORBOX);
    CREATE_NO_WINDOW = 0x08000000    # From Windows API
    subprocess_flags = CREATE_NO_WINDOW
else:
    subprocess_flags = 0

gpucard = opt.gpu_setting
benchmark_cfg = "configs/gpus/%s.cfg" % gpucard
dl_cfg = "configs/benchmarks/%s.cfg" % opt.dl_setting

APP_ROOT = 'applications'
LOG_ROOT = 'logs/%s/%s' % (gpucard, conv_algo)

if not os.path.exists(LOG_ROOT):
    os.makedirs(LOG_ROOT)

# Reading benchmark settings
cf_bs = ConfigParser.SafeConfigParser()
cf_bs.read(benchmark_cfg)

running_iters = cf_bs.getint("profile_control", "iters")
#running_time = cf_bs.getint("profile_control", "secs")
nvIns_dev_id = cf_bs.getint("profile_control", "nvIns_device_id")
cuda_dev_id = cf_bs.getint("profile_control", "cuda_device_id")
pw_sample_int = cf_bs.getint("profile_control", "power_sample_interval")
rest_int = cf_bs.getint("profile_control", "rest_time")
#metrics = json.loads(cf_bs.get("profile_control", "metrics"))
core_frequencies = json.loads(cf_bs.get("dvfs_control", "coreF"))
memory_frequencies = json.loads(cf_bs.get("dvfs_control", "memF"))
powerState = cf_bs.getint("dvfs_control", "powerState")
if powerState == 5:
    freqState = 1
else:
    freqState = powerState

# Read GPU application settings
cf_ks = ConfigParser.SafeConfigParser()
cf_ks.read(dl_cfg)
benchmark_programs = cf_ks.sections()

print benchmark_programs
#print metrics
print core_frequencies
print memory_frequencies

if 'linux' in sys.platform:
    pw_sampling_cmd = 'nohup ./nvml_samples -device=%d -si=%d -output=%s/%s 1>null 2>&1 &'
    app_exec_cmd = './%s/%s %s 1>>%s/%s 2>&1'
    dvfs_cmd = 'gpu=%d fcore=%s fmem=%s ./adjustClock.sh' % (nvIns_dev_id, '%s', '%s')
    kill_pw_cmd = 'killall nvml_samples'
elif 'win' in sys.platform:
    APP_ROOT=''
    pw_sampling_cmd = 'start /B nvml_samples.exe -device=%d -si=%d -output=%s/%s > nul'
    #app_exec_cmd = '%s\\%s %s -device=%d -secs=%d >> %s/%s'
    app_exec_cmd = '%s\\%s %s >> %s/%s 2>&1' # for win caffe
    #dvfs_cmd = 'nvidiaInspector.exe -forcepstate:%s,%s -setMemoryClock:%s,1,%s -setGpuClock:%s,1,%s'
    if powerState !=0:
        dvfs_cmd = 'nvidiaInspector.exe -forcepstate:%s,%d -setGpuClock:%s,%d,%s -setMemoryClock:%s,%d,%s' % (nvIns_dev_id, powerState, nvIns_dev_id, freqState, '%s', nvIns_dev_id, freqState, '%s')
    else:
        dvfs_cmd = 'nvidiaInspector.exe -setBaseClockOffset:%s,%d,%s -setMemoryClockOffset:%s,%d,%s' % (nvIns_dev_id, freqState, '%s', nvIns_dev_id, freqState, '%s')
    kill_pw_cmd = 'tasklist|findstr "nvml_samples.exe" && taskkill /F /IM nvml_samples.exe'

for core_f in core_frequencies:
    for mem_f in memory_frequencies:

        # set specific frequency
        command = dvfs_cmd % (core_f, mem_f)
        
        print command
        os.system(command)
        time.sleep(rest_int)

        for app in benchmark_programs:

            args = json.loads(cf_ks.get(app, 'args'))
            #train_path = cf_ks.get(app, 'train_data')
            #test_path = cf_ks.get(app, 'test_data')

            #argNo = 0

            for arg in args:
                network, batch_size = arg.split()
                arg_name = "-".join(arg.split())
                # arg, number = re.subn('-device=[0-9]*', '-device=%d' % cuda_dev_id, arg)
                powerlog = 'benchmark_%s_%s_core%d_mem%d_power.log' % (app, arg_name, core_f, mem_f)
                perflog = 'benchmark_%s_%s_core%d_mem%d_perf.log' % (app, arg_name, core_f, mem_f)
                metricslog = 'benchmark_%s_%s_core%d_mem%d_metrics.log' % (app, arg_name, core_f, mem_f)


                # start record power data
                os.system("echo \"app:%s,arg:%s\" > %s/%s" % (app, arg_name, LOG_ROOT, powerlog))
                command = pw_sampling_cmd % (nvIns_dev_id, pw_sample_int, LOG_ROOT, powerlog)
                print command
                os.system(command)
                time.sleep(rest_int)

                ## set data path for the network
                #def set_datapath(network):
                #    network_path = "networks/%s.prototxt" % network

		#    replacement_list = {
		#        '$TRAIN_PATH': ('%s' % train_path),
		#        '$TEST_PATH': ('%s' % test_path),
		#    }

		#    proto = ''
		#    tfile = open(network_path, "r")
		#    proto = tfile.read()
		#    tfile.close()

		#    for r in replacement_list:
		#        proto = proto.replace(r, replacement_list[r])
		#    
		#    tfile = open('tmp/%s.prototxt' % network, "w")
		#    tfile.write(proto)
		#    tfile.close()
                #    
                #set_datapath(arg)    
                if app == 'caffe':
                    exec_arg = "time -model tmp/%s-%s.prototxt -gpu %d -iterations %d" % (network, batch_size, cuda_dev_id, running_iters)
                if app == 'giexec':
                    exec_arg = "--model=./models/%s/%s.caffemodel --deploy=./models/%s/deploy.prototxt --output=prob --batch=%s --device=%d --iterations=%d" % (network, network, network, batch_size, cuda_dev_id, running_iters / 10)
                # execute program to collect power data
                os.system("echo \"app:%s,arg:%s\" > %s/%s" % (app, arg, LOG_ROOT, perflog))
                command = app_exec_cmd % (APP_ROOT, app, exec_arg, LOG_ROOT, perflog)
                #command = app_exec_cmd % (app, exec_arg, LOG_ROOT, perflog)  # for win caffe
                print command
                #os.system(command)
                p = subprocess.Popen(command, shell=True)
                p.wait()
                #r_stdout = subprocess.Popen(command,
                #            stdout=subprocess.PIPE,
                #            stderr=subprocess.PIPE,
                #            creationflags=subprocess_flags).communicate()[1]
                time.sleep(rest_int)

                # stop record power data
                os.system(kill_pw_cmd)

                ## execute program to collect time data
                ##command = 'nvprof --profile-child-processes %s/%s %s >> %s/%s 2>&1' % (APP_ROOT, app, exec_arg, LOG_ROOT, perflog)
                #command = 'nvprof --profile-child-processes %s %s >> %s/%s 2>&1' % (app, exec_arg, LOG_ROOT, perflog) # for win caffe
                #print command
                #os.system(command)
                #time.sleep(rest_int)

                ## collect grid and block settings
                ##command = 'nvprof --print-gpu-trace --profile-child-processes %s/%s %s  > %s/%s 2>&1' % (APP_ROOT, app, exec_arg, LOG_ROOT, metricslog)
                #command = 'nvprof --print-gpu-trace --profile-child-processes %s %s  > %s/%s 2>&1' % (app, exec_arg, LOG_ROOT, metricslog) # for win caffe
                #print command
                #os.system(command)
                #time.sleep(rest_int)

                ## execute program to collect metrics data
                #metCount = 0

                ## to be fixed, the stride should be a multiplier of the metric number
                #while metCount < len(metrics):

                #    if metCount + 3 > len(metrics):
                #        metStr = ','.join(metrics[metCount:])
                #    else:
                #        metStr = ','.join(metrics[metCount:metCount + 3])
                #    command = 'nvprof --devices %s --metrics %s %s/%s %s -device=%d -iters=50 >> %s/%s 2>&1' % (cuda_dev_id, metStr, APP_ROOT, app, arg, cuda_dev_id, LOG_ROOT, metricslog)
                #    print command
                #    os.system(command)
                #    time.sleep(rest_int)
                #    metCount += 3
		#argNo += 1

time.sleep(rest_int)
