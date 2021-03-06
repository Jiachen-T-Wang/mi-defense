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

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

device = "cuda"
dataset_name = "facescrub"
root_path = "/opt/tiger/vehiclereid/result/align"
save_model_path = os.path.join(root_path, "target_models")
save_log_path = os.path.join(root_path, "target_logs")
save_img_path = os.path.join(root_path, "target_imgs")
os.makedirs(save_img_path, exist_ok=True)
os.makedirs(save_log_path, exist_ok=True)
os.makedirs(save_model_path, exist_ok=True)

model_name = "SimpleCNN"
model_v = "dp2"
n_epochs = 300
noise = 0.05
dp_lr = 1e-2
momentum = 0.9
weight_decay = 1e-4

microbatch_size = 1
g_clip_thres = 1

def eval_net(net, dataloader):
    net.eval()
    cnt, acc, loss_tot = 0, 0, 0
    for img, label in dataloader:
        img, label = img.to(device), label.to(device)
        label = label.view(-1)
        out_prob = net(img)
        out_label = torch.argmax(out_prob, dim=1).view(-1)
        acc += torch.sum(out_label == label).item()
        cnt += img.size(0)

    return acc * 100.0 / cnt

def main(args, trainloader, testloader, noise_multiplier):
    net = utils.get_model(args, model_name, model_v)
    net = torch.nn.DataParallel(net).to(device)
    
    optimizer = torch.optim.SGD(net.parameters(), 
                                lr=dp_lr, 
                                momentum=momentum, 
                                weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss(reduction='none').cuda()
    best_ACC = 0
    
    print("Start Training!")
    for e in range(n_epochs):
        tf = time.time()
        net.train()
        loss_tot, avg_norm, cnt, step = 0.0, 0.0, 0, 0

        for imgs, labels in trainloader:
            imgs, labels = imgs.to(device), labels.to(device)
            bs = imgs.size(0)
            outputs = net(imgs)
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
        train_accuracy = eval_net(net, trainloader)
        test_accuracy = eval_net(net, testloader)
        interval = time.time() - tf
        print("Epoch:{}\tTime:{:.2f}\tTrain Loss:{:.2f}\tTrain Acc:{:.2f}\tTest Acc:{:.2f}".format(e, interval, train_loss, train_accuracy, test_accuracy))
        
        if test_accuracy > best_ACC:
            best_ACC = test_accuracy
            best_model = deepcopy(net)
            torch.save({'state_dict':best_model.state_dict()}, os.path.join(save_model_path, "{}_{}.tar".format(model_name, model_v)))
    
    print("The best accuracy is %.03f." % (best_ACC))
    print("==============================================================================")
    
if __name__ == '__main__':
    file = dataset_name + ".json"
    args = utils.load_params(file)
    log_file = "train_" + args['dataset']['model_name'] + '_{}.txt'.format(model_v)
    logger = utils.Tee(os.path.join(save_log_path, log_file), 'w')

    train_file = args['dataset']['train_file']
    test_file = args['dataset']['test_file']
    trainloader = utils.init_dataloader(args, train_file, mode="train")
    testloader = utils.init_dataloader(args, test_file, mode="test")
    main(args, trainloader, testloader, noise)
