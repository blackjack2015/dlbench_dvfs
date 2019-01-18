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
parser.add_argument('--batch_size', type=int, default=20)
parser.add_argument('--num_epochs', type=int, default=40)
parser.add_argument('--dp_keep_prob', type=float, default=0.35)
parser.add_argument('--inital_lr', type=float, default=1.0)
parser.add_argument('--save', type=str, default='lstm_model.pt')
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

def run_epoch(model, data_loader, batch_size, is_train=False, lr=0.1):
    """Runs the model on the given data."""
    hidden = model.init_hidden()
    start_time = time.time()
    costs = 0.0
    iters = 0
    if is_train:
        model.train()
    else:
        model.eval()
    for step, (x, y) in enumerate(data_loader):
        #print(x)
        #print(type(x))
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
            if step % (epoch_size // 10) == 10:
                logging.info("{} perplexity: {:8.2f}".format(step * 1.0 / epoch_size, np.exp(costs / iters)))
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
