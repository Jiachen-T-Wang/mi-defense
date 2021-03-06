import time
import torch
import numpy as np
from utils import *
from copy import deepcopy
import torch.nn.functional as F

device = "cuda"

def eval_net(net, testloader, classes=10):
    total = 0
    correct = 0
    with torch.no_grad():
        net.eval()
        for i, (imgs, lbls) in enumerate(testloader):

            imgs, lbls = imgs.to(device), lbls.to(device)
            output = net(imgs)[0]
            predicted = output.argmax(dim=1)

            total += imgs.size(0)
            correct += predicted.eq(lbls).sum().item()

    accuracy = 100*(correct/total)
    
    return accuracy

def train_vib(net, train_loader, test_loader, optimizer, criterion, epoch, classes=10, beta=1e-2):
    best_ACC = 0.0
    best_model = None
    #for epoch in range(n_epochs):
    tf = time.time()
    net.train()
    loss_tot, cnt = 0.0, 0
    for i, batch in enumerate(train_loader):

        imgs, labels = batch
        imgs, labels = imgs.to(device), labels.to(device)

        outputs, mu, std = net(imgs)

        cross_loss = criterion(outputs, labels)
        # loss = -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        info_loss = - 0.5 * (1 + 2 * std.log() - mu.pow(2) - std.pow(2)).sum(dim=1).mean()
        loss = cross_loss + beta * info_loss 
            
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_tot += loss.item() * imgs.shape[0]
        cnt += imgs.shape[0]
            
        
    train_loss = loss_tot / cnt
    train_accuracy = eval_net(net, train_loader, classes)
    test_accuracy = eval_net(net, test_loader, classes)
    interval = time.time() - tf
    #if test_accuracy > best_ACC:
    #best_ACC = test_accuracy
    #best_model = deepcopy(net)

    print("Epoch:{}\tTime:{:.2f}\tTrain Loss:{:.2f}\tTrain Acc:{:.2f}\tTest Acc:{:.2f}".format(epoch, interval, train_loss, train_accuracy, test_accuracy))

    #print("Best Acc:{:.2f}".format(best_ACC))
    return test_accuracy


def train(net, train_loader, test_loader, optimizer, criterion, epoch, classes=10):
    losses = []
    best_ACC = 0.0
    best_model = None
    
    tf = time.time()
    net.train()
    loss_tot, cnt = 0.0, 0
    for i, batch in enumerate(train_loader):
        imgs, labels = batch
        imgs, labels = imgs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = net(imgs)[0]

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        loss_tot += loss.item() * imgs.shape[0]
        cnt += imgs.shape[0]
        losses.append(loss.item())

    train_loss = loss_tot / cnt
    train_accuracy = eval_net(net, train_loader, classes)
    test_accuracy = eval_net(net, test_loader, classes)
    interval = time.time() - tf


    print("Epoch:{}\tTime:{:.2f}\tTrain Loss:{:.2f}\tTrain Acc:{:.2f}\tTest Acc:{:.2f}".format(epoch, interval, train_loss, train_accuracy, test_accuracy))

    return test_accuracy




