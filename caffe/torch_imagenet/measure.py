import argparse
import os
import random
import shutil
import time
import warnings
import sys
import logging
import GPUtil


from threading import Thread

class Measure:

    def __init__(self):
        self.reset()
        self.ifmeasure = False

    def reset(self):
         # io time
        self.io_time = GapMeter()
        # h2d_time 
        self.h2d_time = GapMeter()
        # gpu_time
        self.gpu_time = GapMeter()
        # batch_time
        self.batch_time = GapMeter()

    def add_GPUmonitor(self, delay):
        self.GPUmonitor = GPUMonitor(delay)
        # gpu_load
        self.gpu_load = AverageMeter()
        # gpu_load_record
        # self.gpu_load_record = AverageMeter()
        # gpu_speed
        self.gpu_speed = AverageMeter()

    def tomeasure(self):
        self.ifmeasure = True

    def __getattr__(self, attr, val):
        if not self.ifmeasure:
            raise ValueError('{} not measure mode ')


class GapMeter(object):
    """Computes and stores the average and current value"""	
    def __init__(self):
    	self.reset()

    def reset(self):
        self.start = 0
        self.end = 0
        self.gap = 0
        self.avemeter = AverageMeter()
        self.metering = False

    def update_start(self, start):
        self.start = start
        self.metering = True

    def update_end(self, end):
        try:
            if self.metering:
                self.end = end
                self.gap = self.end - self.start
                self.avemeter.update(self.gap)
                self.metering = False
            else:
                raise RuntimeError('not metering')
        except:
            print('========please start to            meter before end it ==============')    		


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class GPUMonitor(Thread):
    def __init__(self, delay):
        super(GPUMonitor, self).__init__()
        self.stopped = False
        self.delay = delay # Time between calls to GPUtil
        self.start()
        self.GPUs = GPUtil.getGPUs()

    def getInfo():
        return [(self.GPUs[i].load, self.GPUs[i].memoryUtil, self.GPUs[i].memoryUsed)
            for i in range(len(GPUs))]

    def getLoad():
        return [self.GPUs[i].load for i in range(len(GPUs))]

    def run(self):
        while not self.stopped:
            self.GPUs = GPUtil.getGPUs()
            time.sleep(self.delay)

    def stop(self):
        self.stopped = True
