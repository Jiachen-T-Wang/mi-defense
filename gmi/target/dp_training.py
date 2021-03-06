import torch, os, time, collections, model
import numpy as np 
import pandas as pd
import torch.nn as nn
import utils
from copy import deepcopy
from rdp_accountant import compute_rdp, get_privacy_spent

device = "cuda"
microbatch_size = 1
delta = 1e-5

def train_dp(model_name, model_path, mode, model, criterion, optimizer, trainloader, testloader, n_epochs, noise_multiplier, g_clip_thres=1):
    step = 0
    best_ACC = 0.0
    best_eps = 0.0
    for epoch in range(n_epochs):
        model.train()
        tf = time.time()
        avg_norm, cnt = 0.0, 0

        for i, (img, iden) in enumerate(trainloader):
            img, iden = img.to(device), iden.to(device)
            bs = img.size(0)
            iden = iden.view(-1)
            
            __, out_prob = model(img)
            loss = criterion(out_prob, iden)
            loss_val = torch.mean(loss).item()
            grad = [torch.zeros_like(param) for param in model.parameters()]
            num_microbatch = bs / microbatch_size
            for j in range(0, bs, microbatch_size):
                optimizer.zero_grad()
                torch.autograd.backward(torch.mean(loss[j:j+microbatch_size]), retain_graph=True)

                L2_norm = 0.0
                for param in model.parameters():
                    L2_norm += (param.grad * param.grad).sum()
                L2_norm = torch.sqrt(L2_norm)
                avg_norm = avg_norm * 0.95 + L2_norm * 0.05

                coef = float(g_clip_thres) / max(g_clip_thres, L2_norm.item())
                grad = [g + param.grad * coef for param, g in zip(model.parameters(), grad)]

            for param, g in zip(model.parameters(), grad):
                if noise_multiplier > 0:
                    param.grad.data = g + torch.cuda.FloatTensor(g.size()).normal_(0, noise_multiplier * float(g_clip_thres))
                else:
                    param.grad.data = g
                param.grad.data /= num_microbatch

            optimizer.step()
            cnt += bs
            step += 1
            
           


        train_time = time.time() - tf
        train_loss, train_acc = test(model, criterion, trainloader)
        test_loss, test_acc = test(model, criterion, testloader)

        '''
        q = batch_size * 1.0 / cnt
        orders = [1 + x / 10. for x in range(1, 100)] + list(range(12, 256))

        if noise_multiplier > 0:
            rdp = compute_rdp(q=q, noise_multiplier=noise_multiplier, steps=step, orders=orders)
            eps = get_privacy_spent(orders, rdp, target_delta=delta)[0]
        else:
            eps = 0
        '''

        interval = time.time() - tf
        if test_acc > best_ACC:
            best_ACC = test_acc
            best_model = deepcopy(model)
            torch.save({'state_dict':best_model.state_dict()}, os.path.join(model_path, "{}_{}_{:.2f}.tar").format(model_name, mode, best_ACC))
            

        print("Epoch:{}\tTime:{:.2f}\tTrain Loss:{:.2f}\tTrain Acc:{:.2f}\tTest Acc:{:.2f}".format(epoch, interval, train_loss, train_acc, test_acc))
        

    print("Best Acc:{:.2f}".format(best_ACC))
   
    



def test(model, criterion, dataloader):
    tf = time.time()
    model.eval()
    loss, cnt, ACC = 0.0, 0, 0

    for i, (img, iden) in enumerate(dataloader):
        img, iden = img.to(device), iden.to(device)
        bs = img.size(0)
        iden = iden.view(-1)

        ___, out_prob = model(img)
        cross_loss = criterion(out_prob, iden)
        out_iden = torch.argmax(out_prob, dim=1).view(-1)
        ACC += torch.sum(iden == out_iden).item()
        loss += torch.sum(cross_loss).item()
        cnt += bs

    return loss * 1.0 / cnt, ACC * 100.0 / cnt




