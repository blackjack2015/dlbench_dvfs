import argparse
import os
import random
import shutil
import time
import warnings
import sys
import logging
import measure
import getlogger
import gc

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

from datasets import DatasetHDF5  
from networks.alexnet import AlexNet
from pprint import pformat

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--customize', action='store_true',default=False,
                    help='choose model from pytorch or self-defined')
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=95, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=64, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--optim', default='SGD', type=str, metavar='OPTIM',
                    help='choose optimizer')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--world-size', default=1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=0, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',default=False,
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')
parser.add_argument('--kind', default='0', type=str, metavar='N',
                    help='number of adjustment (default: 0)')
parser.add_argument('--measure', type=str, default=None,
                    help=' if mode of measurement')
parser.add_argument('--iterations', type=int, default=20,
                    help=' how many iterations in the mode of measurement')

best_acc1 = 0
meas1 = measure.Measure()


def main2():
    args = parser.parse_args()

    if not os.path.exists('log'):
        os.mkdir('log')

    if args.customize:
        logging_name = 'log' + '_self_' + args.arch + '_'+ args.optim + args.kind + '.txt' 
    else:
        logging_name = 'log' + '_default_' + args.arch  + '_' + args.optim + args.kind + '.txt'

    logging_path = os.path.join('log', logging_name) 

    logging.basicConfig(level=logging.DEBUG,
                        filename=logging_path,
                        filemode='a',
                        format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s',
                        datefmt='%H:%M:%S')

    logging.debug('this is a logging debug message')
    logging.info('Logging for {}'.format(args.arch))
    logging.info('optim : [{}], batch_size = {}, lr = {}, weight_decay= {}, momentum = {}'.format( \
                    args.optim, args.batch_size,
                    args.lr, args.weight_decay, args.momentum) )


    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    # if args.multiprocessing_distributed:
    #     # Since we have ngpus_per_node processes per node, the total world_size
    #     # needs to be adjusted accordingly
    #     args.world_size = ngpus_per_node * args.world_size
    #     # Use torch.multiprocessing.spawn to launch distributed processes: the
    #     # main_worker process function
    #     mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    # else:
    #     # Simply call main_worker function
    #     main_worker(args.gpu, ngpus_per_node, args)
    
    if args.measure:
        log_path = args.measure
        if not os.path.exists(log_path):
            os.mkdir(log_path)
        # batch_sizes = [16, 32, 64, 128, 256]
        # num_workers = [1, 2, 4, 8, 16, 32, 64]
        # for batch_sizei in batch_sizes:
        #     for num_workeri in num_workers:
        #         args.batch_size = batch_sizei
        #         args.workers = num_workeri
        main_worker(args.gpu, ngpus_per_node, args)
    return
        


def main_worker(gpu, ngpus_per_node, args):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode


    end = time.time()

    global meas1
    global best_acc1
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.measure:
        meas1.add_GPUmonitor(0.05)
        meas1.tomeasure()
    
        log_path = args.measure
        if not os.path.exists(log_path):
            os.mkdir(log_path)
        # batch_sizes = [16, 32, 64, 128, 256]
        # num_workers = [1, 2, 4, 8, 16, 32, 64]
        # for batch_sizei in batch_sizes:
        #     for num_workeri in num_workers:
    

        # create model
        if args.pretrained:
            print("=> using pre-trained model '{}'".format(args.arch))
            model = models.__dict__[args.arch](pretrained=True)
        else:
            print("=> creating model '{}'".format(args.arch))
            if args.customize:
                print("=> self-defined model '{}'".format(args.arch))
                model = AlexNet()
            else:
                model = models.__dict__[args.arch]()
        
        model.train()
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model = model.cuda(args.gpu)
        else:
            # DataParallel will divide and allocate batch_size to all available GPUs
            if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
                # model.features = torch.nn.DataParallel(model.features)
                model.cuda()
            else:
                model = torch.nn.DataParallel(model).cuda()

        # define loss function (criterion) and optimizer
        criterion = nn.CrossEntropyLoss().cuda(args.gpu)

        if args.optim == 'SGD':
            optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                        momentum=args.momentum,
                                        weight_decay=args.weight_decay)
        elif args.optim == 'Adam':
            optimizer = torch.optim.Adam(model.parameters(),
                                         lr=args.lr, betas=(0.9, 0.999),
                                         weight_decay=args.weight_decay)

        cudnn.benchmark = True

        # Data loading code
        traindir = os.path.join(args.data, 'train')
        valdir = os.path.join(args.data, 'val')
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        # trainset = torchvision.datasets.ImageFolder(traindir, transforms.Compose([
        # trainset = datasets.ImageFolder(
        #     traindir,
        #     transforms.Compose([
        #         transforms.RandomResizedCrop(224),
        #         transforms.RandomHorizontalFlip(),
        #         transforms.ToTensor(),
        #         normalize,
        #     ]))

        # hdf5fn = os.path.join(args.data, 'imagenet-shuffled.hdf5')

        if args.customize:
            size_resize = 227
        else:
            size_resize = 224
        
        log_name = args.arch + meas1.GPUmonitor.GPUs[args.gpu].name + 'b' + str(args.batch_size) \
                   + 'n' + str(args.workers) + '.log'
        log_file = os.path.join(log_path, log_name)
    
        logger1 = getlogger.get_logger('measure', log_file)
    
        logger1.info('>>>==================')
    
        logger1.info('***batch_size: *[{}]*, num_workers: *[{}]*")'.
                     format(args.batch_size, args.workers))
        one_measure(args, meas1, logger1,
                    args.batch_size, args.workers, model, criterion, optimizer, size_resize, args.iterations)
        del model
        gc.collect()
        for i in range(8):
            print('wait %d seconds for collecting unused memory', i)
            time.sleep(1)
        meas1.reset()
        meas1.gpu_load.reset()
        meas1.gpu_speed.reset()
        meas1.GPUmonitor.stop()
        return

    if args.distributed:
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
    # create model
    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))
        model = models.__dict__[args.arch](pretrained=True)
    else:
        print("=> creating model '{}'".format(args.arch))
        if args.customize:
            print("=> self-defined model '{}'".format(args.arch))
            model = AlexNet()
            model.apply(weights_init)
            print('model initialized')
        else:
            model = models.__dict__[args.arch]()
            if args.arch == 'alexnet' :
                model.apply(weights_init)
            print('model initialized')
    model.train()
    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            #model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)


    if args.optim == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)
    elif args.optim == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), 
                                    lr=args.lr, betas=(0.9,0.999),
                                    weight_decay=args.weight_decay)

    # optionally resume from a checkpoint

    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # Data loading code
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    #trainset = torchvision.datasets.ImageFolder(traindir, transforms.Compose([
    # trainset = datasets.ImageFolder(
    #     traindir,
    #     transforms.Compose([
    #         transforms.RandomResizedCrop(224),
    #         transforms.RandomHorizontalFlip(),
    #         transforms.ToTensor(),
    #         normalize,
    #     ]))

    hdf5fn = os.path.join(args.data, 'imagenet-shuffled.hdf5')

    if args.customize:
        size_resize = 227
    else:
        size_resize = 224

    trainset = DatasetHDF5(hdf5fn, 'train', transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomResizedCrop(size_resize),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
        ]))

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(trainset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        trainset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    testset = DatasetHDF5(hdf5fn, 'val', transforms.Compose([
            transforms.ToPILImage(),
    #        transforms.Scale(256),
            transforms.CenterCrop(size_resize),
            transforms.ToTensor(),
            normalize,
        ]))
    val_loader = torch.utils.data.DataLoader(
        testset,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    # val_loader = torch.utils.data.DataLoader(
    #     datasets.ImageFolder(valdir, transforms.Compose([
    #         transforms.Resize(256),
    #         transforms.CenterCrop(224),
    #         transforms.ToTensor(),
    #         normalize,
    #     ])),
    #     batch_size=args.batch_size, shuffle=False,
    #     num_workers=args.workers, pin_memory=True)



    if args.evaluate:
        validate(val_loader, model, criterion)
        return

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, args)

        # evaluate on validation set
        acc1 = validate(val_loader, model, criterion, args)

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_acc1': best_acc1,
            'optimizer' : optimizer.state_dict(),
        }, is_best)
    torch.save(model.state_dict(), args.arch + args.optim + args.kind + 'params.pth')

def one_measure(args, meas1, logger1, batch_size, num_workers, model, criterion, optimizer, size_resize, iterations):
    """
        measure one set of args batch_siez and num_workers
    """
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    hdf5fn = os.path.join(args.data, 'imagenet-shuffled.hdf5')
    trainset = DatasetHDF5(hdf5fn, 'train', transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomResizedCrop(size_resize),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
        ]))

    train_loader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True)

    epoch = 0
    #for epoch in range(2):
        # train for measuring

    train_iter = iter(train_loader)

    meas1.io_time.update_start(time.time())
    meas1.batch_time.update_start(time.time())
    i = 0
    for input, target in train_iter:

        gpu_load_records = []
        # watch gpu_load
        meas1.gpu_load.update(meas1.GPUmonitor.GPUs[args.gpu].load)
        gpu_load_records.append(meas1.gpu_load.val)

        # measure data loading times
        meas1.io_time.update_end(time.time())

        # watch gpu_load
        meas1.gpu_load.update(meas1.GPUmonitor.GPUs[args.gpu].load)
        gpu_load_records.append(meas1.gpu_load.val)

        # measure data loaded to GPU time
        meas1.h2d_time.update_start(time.time())
        if args.gpu is not None:
            input = input.cuda(args.gpu, non_blocking=True)
        target = target.cuda(args.gpu, non_blocking=True)
        meas1.h2d_time.update_end(time.time())

        # watch gpu_load
        meas1.gpu_load.update(meas1.GPUmonitor.GPUs[args.gpu].load)
        gpu_load_records.append(meas1.gpu_load.val)

        # compute output
        meas1.gpu_time.update_start(time.time())
        output = model(input)
        loss = criterion(output, target)

        # watch gpu_load
        meas1.gpu_load.update(meas1.GPUmonitor.GPUs[args.gpu].load)
        gpu_load_records.append(meas1.gpu_load.val)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        meas1.gpu_time.update_end(time.time())

        # calculate gpu_speed
        meas1.gpu_speed.update( 1 / ( meas1.gpu_time.gap / batch_size ))

        # watch gpu_load
        meas1.gpu_load.update(meas1.GPUmonitor.GPUs[args.gpu].load)
        gpu_load_records.append(meas1.gpu_load.val)

        print(' train* ===Epoch: [{0}][{1}/{2}]\t'
          .format(epoch, i, len(train_loader)))
        print('>>> *io_time : [{}] *h2d_time :[{}] gpu_time :[{}] '
                     '  *batch_time :[{}] *gpu_speed :[{}] image/s'
                     ' **gpu_load :[{},{},{},{},{}]'
                    .format(meas1.io_time.gap, meas1.h2d_time.gap, meas1.gpu_time.gap,
                    meas1.batch_time.gap, meas1.gpu_speed.val, gpu_load_records[0],
                    gpu_load_records[1], gpu_load_records[2], gpu_load_records[3],
                    gpu_load_records[4]
                     ))
        logger1.info('>>> ===========********  measing batch ===========')
        logger1.info(' train* ===Epoch: [{0}][{1}/{2}]\t'
          .format(epoch, i, len(train_loader)))

        meas1.io_time.update_start(time.time())
        meas1.batch_time.update_end(time.time())
        logger1.info('>>> *io_time : [{}] *h2d_time :[{}] gpu_time :[{}] '
                     '  *batch_time :[{}] *gpu_speed :[{}] image/s'
                     ' **gpu_load :[{},{},{},{},{}]'
                    .format(meas1.io_time.gap, meas1.h2d_time.gap, meas1.gpu_time.gap,
                    meas1.batch_time.gap, meas1.gpu_speed.val, gpu_load_records[0],
                    gpu_load_records[1], gpu_load_records[2], gpu_load_records[3],
                    gpu_load_records[4]
                     ))
        meas1.batch_time.update_start(time.time())
        if i == iterations:
            for indexqueue in train_iter.index_queues:
                while not indexqueue.empty():
                    indexqueue.get()
            while not train_iter.worker_result_queue.empty():
                train_iter.worker_result_queue.get()
            for process in train_iter.workers:
                process.terminate()
                process.join()
            del input
            del target
            break
        i += 1

    gc.collect()
    logger1.info('>>>average time  ========= **io_time : [{}] **h2d_time :[{}] '
         '**gpu_time :[{}] **batch_time :[{}] '
         '**gpu_speed :[{} image/s] **gpu_load :[{}]'
        .format(meas1.io_time.avemeter.avg, meas1.h2d_time.avemeter.avg,
            meas1.gpu_time.avemeter.avg, meas1.batch_time.avemeter.avg,
            meas1.gpu_speed.avg, meas1.gpu_load.avg
         ))

def train(train_loader, model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    disk_time = AverageMeter()


    # switch to train mode
    model.train()


    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading times
        data_time.update(time.time() - end)

        if args.gpu is not None:
            input = input.cuda(args.gpu, non_blocking=True)
        target = target.cuda(args.gpu, non_blocking=True)

        # compute output
        output = model(input)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(acc1[0], input.size(0))
        top5.update(acc5[0], input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1, top5=top5))
            logging.info(' train* ===Epoch: [{0}][{1}/{2}]\t Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f} Loss {loss.val:.4f}'
              .format(epoch, i, len(train_loader), top1=top1, top5=top5, loss=losses))


def validate(val_loader, model, criterion, args):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(val_loader):
            if args.gpu is not None:
                input = input.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(acc1[0], input.size(0))
            top5.update(acc5[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                       i, len(val_loader), batch_time=batch_time, loss=losses,
                       top1=top1, top5=top5))

        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))
        logging.info(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f} Loss {loss.val:.4f}'
              .format(top1=top1, top5=top5, loss=losses))

    return top1.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')

# torch.save(model_object.state_dict(), 'params.pth')  
# model_object.load_state_dict(torch.load('params.pth'))  

def weights_init(m):
    """ init weights of net   """
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight.data, mean=0, std=0.01)
        nn.init.normal_(m.bias.data, mean=0, std=0.01)
    if isinstance(m, nn.Conv2d):
        nn.init.normal_(m.weight.data, mean=0, std=0.01)
        nn.init.normal_(m.bias.data, mean=0, std=0.01)
    if isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
        
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


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res
    
main2()
sys.exit()