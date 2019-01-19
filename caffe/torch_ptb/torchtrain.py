import argparse
import time
import os
import sys
import torch
import torch.nn
from torch.autograd import Variable
from torch.utils import data as data_
import torch.utils.data.distributed
import torch.nn as nn
from lm import repackage_hidden, lstm
import treader
import numpy as np
import logging

logging.getLogger().setLevel(logging.DEBUG)

parser = argparse.ArgumentParser(description='LSTM-PTB')
parser.add_argument('--data', type=str, default='data')
parser.add_argument('--hidden_size', type=int, default=1500)
parser.add_argument('--num_steps', type=int, default=35)
parser.add_argument('--num_layers', type=int, default=2)
parser.add_argument('-b', '--batch_size', type=int, default=20)
parser.add_argument('--num_epochs', type=int, default=40)
parser.add_argument('--dp_keep_prob', type=float, default=0.35)
parser.add_argument('--inital_lr', type=float, default=1.0)
parser.add_argument('--save', type=str, default='lstm_model.pt')

parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--measure', type=str, default=None,
                    help=' if mode of measurement')
parser.add_argument('--iterations', type=int, default=20,
                    help=' how many iterations in the mode of measurement')
parser.add_argument('-t', '--run-time', type=int, default=None,
                    help=' how many seconds in the mode of measurement')

args = parser.parse_args()

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
criterion = nn.CrossEntropyLoss()

def initLogging(logFilename):
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s-%(levelname)s-%(message)s',
        datefmt='%y-%m-%d %H:%M',
        filename=logFilename,
        filemode='w');
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s-%(levelname)s-%(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

def run_epoch_measure(model, data_loader, batch_size, args, run_time, iterations=200, is_train=True, lr=0.1):
    """Runs the model on the given data."""
    hidden = model.init_hidden()
    costs = 0.0
    iters = 0
    criterion_measure = nn.CrossEntropyLoss().cuda(args.gpu)

    if is_train:
        model.train()
    else:
        model.eval()

    stop_in_time = False
    if run_time:
        stop_in_time = True
    batch_time_between_30_40 = 0
    stop_iterations = iterations
    
    batch_time_between_30_40 = 0
    batch_start_time = time.time()
    for i, (x, y) in enumerate(data_loader):
        #print(x)
        #print(type(x))
        # inputs = Variable(torch.from_numpy(x.astype(np.int64)).transpose(0, 1).contiguous()).cuda()
        # if list(x.size())[0] != batch_size:
        #     break
        
        inputs = Variable(x.transpose(0, 1).contiguous()).cuda(args.gpu, non_blocking=True)
        model.zero_grad()
        hidden = repackage_hidden(hidden)
        hidden = hidden.cuda(args.gpu, non_blocking=True)
        outputs, hidden = model(inputs, hidden)
        #  targets = Variable(torch.from_numpy(y.astype(np.int64)).transpose(0, 1).contiguous()).cuda()
        targets = Variable(y.transpose(0, 1).contiguous()).cuda(args.gpu, non_blocking=True)

        tt = torch.squeeze(targets.view(-1, model.batch_size * model.num_steps))
        loss = criterion_measure(outputs.view(-1, model.vocab_size), tt)
        costs += loss.item() * model.num_steps
        iters += model.num_steps
        
        if is_train:
            loss.backward()
            optimizer = torch.optim.SGD(model.parameters(), lr=1)
            torch.nn.utils.clip_grad_norm(model.parameters(), 0.25)
            for p in model.parameters():
                p.data.add_(-lr, p.grad.data)
            if i % (epoch_size // 10) == 10:
                logging.info("{} perplexity: {:8.2f}".format(i * 1.0 / epoch_size, np.exp(costs / iters)))
        
        batch_end_time = time.time()
        batch_time = batch_end_time - batch_start_time

        if (i > 29) & (i < 40):
            batch_time_between_30_40 += batch_time
        if (i == 40) & (stop_in_time):
            stop_iterations = int(run_time / (batch_time_between_30_40 / 10))
            if stop_iterations < 50:
                stop_iterations = 50
            print('==========================================')
            print('==========================================')
            print('==========================================')
            print('==========================================')
            print('=========******adjust  iterations to  *======================')
            print('=========****** %d  *************   =========================' % stop_iterations)
            print('==========================================')
            print('==========================================')
            print('==========================================')
            print('==========================================')

        if i == stop_iterations:
            # for indexqueue in train_iter.index_queues:
            #     while not indexqueue.empty():
            #         indexqueue.get()
            # while not train_iter.worker_result_queue.empty():
            #     train_iter.worker_result_queue.get()
            # for process in train_iter.workers:
            #     process.terminate()
            #     process.join()
            # del input
            # del target
            break

        batch_start_time = time.time()
    return


def run_epoch(model, data_loader, batch_size, is_train=False, lr=0.1):
    """Runs the model on the given data."""
    hidden = model.init_hidden()
    costs = 0.0
    iters = 0
    if is_train:
        model.train()
    else:
        model.eval()
    
    for i, (x, y) in enumerate(data_loader):
        # print(x)
        # print(type(x))
        # inputs = Variable(torch.from_numpy(x.astype(np.int64)).transpose(0, 1).contiguous()).cuda()
        # if list(x.size())[0] != batch_size:
        #     break
        inputs = Variable(x.transpose(0, 1).contiguous()).cuda()
        model.zero_grad()
        hidden = repackage_hidden(hidden)
        outputs, hidden = model(inputs, hidden)
        #  targets = Variable(torch.from_numpy(y.astype(np.int64)).transpose(0, 1).contiguous()).cuda()
        targets = Variable(y.transpose(0, 1).contiguous()).cuda()
        tt = torch.squeeze(targets.view(-1, model.batch_size * model.num_steps))
        loss = criterion(outputs.view(-1, model.vocab_size), tt)
        costs += loss.item() * model.num_steps
        iters += model.num_steps
        
        if is_train:
            loss.backward()
            optimizer = torch.optim.SGD(model.parameters(), lr=1)
            torch.nn.utils.clip_grad_norm(model.parameters(), 0.25)
            for p in model.parameters():
                p.data.add_(-lr, p.grad.data)
            if i % (epoch_size // 10) == 10:
                logging.info("{} perplexity: {:8.2f}".format(i * 1.0 / epoch_size, np.exp(costs / iters)))
    return np.exp(costs / iters)

initLogging('test.log')

if __name__ == "__main__":
    raw_data = treader.ptb_raw_data(data_path=args.data)
    train_data, valid_data, test_data, word_to_id, id_2_word = raw_data
    vocab_size = len(word_to_id)
    print('Vocabluary size: {}'.format(vocab_size))

    print('load data')
    model = lstm(embedding_dim=args.hidden_size, num_steps=args.num_steps, batch_size=args.batch_size,
                 vocab_size=vocab_size, num_layers=args.num_layers, dp_keep_prob=args.dp_keep_prob)

    epoch_size = ((len(train_data) // model.batch_size) - 1) // model.num_steps

    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        model.cuda()
    
    train_set = treader.TrainDataset(train_data, model.batch_size, model.num_steps)
    train_dataloader = data_.DataLoader(train_set, \
                                  batch_size=args.batch_size, \
                                  shuffle=False, \
                                  # pin_memory=True,
                                  num_workers=1)
    
    valid_set = treader.TestDataset(valid_data, model.batch_size, model.num_steps)
    valid_dataloader = data_.DataLoader(valid_set, \
                                  batch_size=args.batch_size, \
                                  shuffle=False, \
                                  # pin_memory=True,
                                  num_workers=1)

    lr = args.inital_lr
    lr_decay_base = 1 / 1.15
    m_flat_lr = 14.0

    if args.measure:
        run_epoch_measure(model, train_dataloader, args.batch_size, args, args.run_time, args.iterations, is_train=True, lr=0.1)
        sys.exit()
    
    logging.info("Training")
    for epoch in range(args.num_epochs):
        lr_decay = lr_decay_base ** max(epoch - m_flat_lr, 0)
        lr = lr * lr_decay  # decay lr if it is time
        train_p = run_epoch(model, train_dataloader, args.batch_size, True, lr)
        logging.info('Train perplexity at epoch {}: {:8.2f}'.format(epoch, train_p))
        logging.info('Validation perplexity at epoch {}: {:8.2f}'.format(epoch, run_epoch(model, valid_dataloader, args.batch_size)))
    logging.info("Testing")
    model.batch_size = 1  # to make sure we process all the data
    logging.info('Test Perplexity: {:8.2f}'.format(run_epoch(model, test_data, args.batch_size)))
    with open(args.save, 'wb') as f:
        torch.save(model, f)
    logging.info("Done")
