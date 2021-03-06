import os
import sys
import time
import utils
import torch
import model
import dataloader
import numpy as np 
import torch.nn as nn
from copy import deepcopy

def train_dp(net, train_loader, test_loader, optimizer, criterion, epoch, noise_multiplier):
    tf = time.time()
    net.train()
    loss_tot, avg_norm, cnt, step = 0.0, 0.0, 0, 0
    for i, batch in enumerate(train_loader):
        imgs, labels = batch
        imgs, labels = imgs.to(device), labels.to(device)
        bs = imgs.size(0)
        outputs = net(imgs)[0]
        loss = criterion(outputs, labels)
        
        grad = [torch.zeros_like(param) for param in net.parameters()]
        num_microbatch = bs / microbatch_size
        for j in range(0, bs, microbatch_size):
            optimizer.zero_grad()
            torch.autograd.backward(loss[j:j+microbatch_size], retain_graph=True)

            L2_norm = 0.0
            for param in net.parameters():
                L2_norm += (param.grad * param.grad).sum()
            L2_norm = torch.sqrt(L2_norm)
            avg_norm = avg_norm * 0.95 + L2_norm * 0.05

            coef = float(g_clip_thres) / max(g_clip_thres, L2_norm.item()) 
            grad = [g + param.grad * coef for param, g in zip(net.parameters(), grad)]

        for param, g in zip(net.parameters(), grad):
            if noise_multiplier > 0:
                param.grad.data = g + torch.cuda.FloatTensor(g.size()).normal_(0, noise_multiplier * float(g_clip_thres))
            else:
                param.grad.data = g
            param.grad.data /= num_microbatch
        
        optimizer.step()
        loss_tot += torch.sum(loss).item()
        cnt += imgs.shape[0]
        step += 1
        
    train_loss = loss_tot / cnt
    
    '''
    q = batch_size * 1.0 / cnt
    orders = [1 + x / 10. for x in range(1, 100)] + list(range(12, 256))
    
    if noise_multiplier > 0:
        rdp = compute_rdp(q=q, noise_multiplier=noise_multiplier, steps=step, orders=orders)
        eps = get_privacy_spent(orders, rdp, target_delta=delta)[0]
    else:
        eps = 0
    '''

    train_accuracy = eval_net(net, train_loader)
    test_accuracy = eval_net(net, test_loader)
    interval = time.time() - tf
    print("Epoch:{}\tTime:{:.2f}\tTrain Loss:{:.2f}\tTrain Acc:{:.2f}\tTest Acc:{:.2f}".format(epoch, interval, train_loss, train_accuracy, test_accuracy))
