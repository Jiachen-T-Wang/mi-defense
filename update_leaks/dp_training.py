import time
import torch
import numpy as np
from utils import *
from copy import deepcopy
import torch.nn.functional as F
from rdp_accountant import compute_rdp, get_privacy_spent

device = "cuda"

microbatch_size = 1
g_clip_thres = 1
delta = 1e-5

def eval_net(net, testloader, classes=10):
    class_correct = np.zeros(classes)
    class_total = np.zeros(classes)
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

            if classes is not None:
                for prediction, lbl in zip(predicted, lbls):

                    class_correct[lbl] += prediction == lbl
                    class_total[lbl] += 1
                    
    accuracy = 100*(correct/total)
    
    return accuracy

def train_dp(net, train_loader, test_loader, optimizer, criterion, epoch, batch_size, classes=10, noise_multiplier=1.1):
    best_ACC = 0.0
    best_eps = 0.0
    step = 0
    best_model = None
    #for epoch in range(n_epochs):
    tf = time.time()
    net.train()
    loss_tot, avg_norm, cnt = 0.0, 0.0, 0
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
    q = batch_size * 1.0 / cnt
    orders = [1 + x / 10. for x in range(1, 100)] + list(range(12, 256))
        
    if noise_multiplier > 0:
        rdp = compute_rdp(q=q, noise_multiplier=noise_multiplier, steps=step, orders=orders)
        eps = get_privacy_spent(orders, rdp, target_delta=delta)[0]
    else:
        eps = 0

    train_accuracy = eval_net(net, train_loader, classes)
    test_accuracy = eval_net(net, test_loader, classes)
    interval = time.time() - tf
    if test_accuracy > best_ACC:
        best_ACC = test_accuracy
        best_model = deepcopy(net)
        best_eps = eps
    
    
    print("Epoch:{}\tTime:{:.2f}\tTrain Loss:{:.2f}\tEps:{:.2f}\tTrain Acc:{:.2f}\tTest Acc:{:.2f}".format(epoch, interval, train_loss, eps, train_accuracy, test_accuracy))

    #print("Best Acc:{:.2f}".format(best_ACC))
    #print("Best Eps:{:.2f}".format(best_eps))
    #return best_model, best_ACC, best_eps
    return test_accuracy



