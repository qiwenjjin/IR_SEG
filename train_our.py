#!/usr/bin/python
# -*- encoding: utf-8 -*-


from logger import setup_logger
from model_our import GMNet
from cityscapes import MF_dataset
from loss import OhemCELoss
from evaluate import evaluate
from optimizer import Optimizer
from lovasz_losses import lovasz_softmax
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.distributed as dist
import numpy as np

import os
import os.path as osp
import logging
import time
import datetime
import argparse


respth = './res'
if not osp.exists(respth): os.makedirs(respth)
logger = logging.getLogger()


def parse_args():
    parse = argparse.ArgumentParser()
    return parse.parse_args()

def _get_batch_label_vector(target, nclass):
        # target is a 3D Variable BxHxW, output is 2D BxnClass
        batch = target.size(0)
        tvect = torch.autograd.Variable(torch.zeros(batch, nclass))
        for i in range(batch):
            hist = torch.histc(target[i].cpu().data.float(), 
                               bins=nclass, min=0,
                               max=nclass-1)
            vect = hist>0
            tvect[i] = vect
        return tvect
    
    
def train():
    args = parse_args()
    setup_logger(respth)
    binary_class_weight = np.array([1.5121, 10.2388])
    binary_class_weight = torch.tensor(binary_class_weight).float().cuda()
    binary_class_weight_vers = np.array([10.2388, 1.5121])
    binary_class_weight_vers = torch.tensor(binary_class_weight_vers).float().cuda()    
    class_weight = np.array(
                [1.5105, 16.6591, 29.4238, 34.6315, 40.0845, 41.4357, 47.9794, 45.3725, 44.9000])
    class_weight = torch.tensor(class_weight).float().cuda()
    binary_class_weight = binary_class_weight.unsqueeze(0)
    binary_class_weight = binary_class_weight.unsqueeze(2)
    binary_class_weight = binary_class_weight.unsqueeze(2)
    binary_class_weight_vers = binary_class_weight_vers.unsqueeze(0)
    binary_class_weight_vers = binary_class_weight_vers.unsqueeze(2)
    binary_class_weight_vers = binary_class_weight_vers.unsqueeze(2)
    ## dataset
    n_classes = 9
    n_img_per_gpu = 4
    n_workers = 1
    cropsize = [1024, 1024]
    ds = MF_dataset('/data/pcl/datasets/irseg', split='train', do_aug=True)
    dl = DataLoader(ds,
                    batch_size = n_img_per_gpu,
                    shuffle = True,
                    num_workers = n_workers,
                    pin_memory = True,
                    drop_last = True)
    criterion = nn.BCELoss()
    criterion.cuda()
    ## model
    ignore_idx = 255
    net = GMNet(n_classes=n_classes)
  #  save_pth = "res/model_final_diss.pth"
  #  net.load_state_dict(torch.load(save_pth))
    net.cuda()
    net.train()

    ## optimizer
    momentum = 0.9
    weight_decay = 5e-4
    lr_start = 1e-2
    max_iter = 80000
    power = 0.9
    warmup_steps = 1000
    warmup_start_lr = 1e-5
    optim = Optimizer(
            model = net,
            lr0 = lr_start,
            momentum = momentum,
            wd = weight_decay,
            warmup_steps = warmup_steps,
            warmup_start_lr = warmup_start_lr,
            max_iter = max_iter,
            power = power)

    ## train loop
    msg_iter = 10
    loss_avg = []
    st = glob_st = time.time()
    diter = iter(dl)
    epoch = 0
    for it in range(max_iter):
        try:
            sample = next(diter)
            if not sample['image'].size()[0]==n_img_per_gpu: raise StopIteration
        except StopIteration:
            epoch += 1
            diter = iter(dl)
            sample = next(diter)
        im = sample['image'].cuda()
        lb = sample['label'].cuda()
        deep = sample['depth'].cuda()
        bd = sample['bound'].cuda()
        bi = sample['binary_label'].cuda()
        H, W = im.size()[2:]
        
        
        optim.zero_grad()
        out,bi_out,bd_out,context_out = net(im,deep)
        se_target = _get_batch_label_vector(lb, nclass=n_classes).type_as(bd_out)
        out1 = F.softmax(out,1)
        bd = F.one_hot(bd)
        bd=bd.permute(0,3,1,2).float()
        bi = F.one_hot(bi)
        bi= bi.permute(0,3,1,2).float()  

        lossp = lovasz_softmax(out1, lb)
        loss_bi = F.binary_cross_entropy_with_logits(bi_out, bi,pos_weight=binary_class_weight)
        loss_bd = F.binary_cross_entropy_with_logits(bd_out, bd) 
        loss_ct = criterion(torch.sigmoid(context_out), se_target) 
        loss = 0.4*lossp+0.2*loss_bd+0.2*loss_bi+0.2*loss_ct
        
        loss.backward()
        optim.step()

        loss_avg.append(loss.item())
        ## print training log message
        if (it+1)%msg_iter==0:
            loss_avg = sum(loss_avg) / len(loss_avg)
            lr = optim.lr
            ed = time.time()
            t_intv, glob_t_intv = ed - st, ed - glob_st
            eta = int((max_iter - it) * (glob_t_intv / it))
            eta = str(datetime.timedelta(seconds=eta))
            msg = ', '.join([
                    'it: {it}/{max_it}',
                    'lr: {lr:4f}',
                    'loss: {loss:.4f}',
                    'eta: {eta}',
                    'time: {time:.4f}',
                ]).format(
                    it = it+1,
                    max_it = max_iter,
                    lr = lr,
                    loss = loss_avg,
                    time = t_intv,
                    eta = eta
                )
            logger.info(msg)
            loss_avg = []
            st = ed
        if it%2000==0:

            checkpoint_model_file = os.path.join('./res/', str(it) + '.pth')
            print('saving check point %s: ' % checkpoint_model_file)
            torch.save(net.state_dict(), checkpoint_model_file)

    ## dump the final model
    save_pth = osp.join(respth, 'model_final.pth')
    torch.save(net.state_dict(), save_pth)
    logger.info('training done, model saved to: {}'.format(save_pth))


if __name__ == "__main__":

    train()
    evaluate()
