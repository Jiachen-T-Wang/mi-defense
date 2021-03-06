import os
import time
import math
import torch
import models
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from utils import load_state_dict, load_data

lr = 1e-3
n_epochs = 50
n_classes = 10
batch_size = 64
device = "cuda"

root_path = "/opt/tiger/vehiclereid/result/update"
model_v = "vib3"
dataset_path = os.path.join(root_path, "multi_dataset/{}".format(model_v))

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

train_data_path = os.path.join(dataset_path, "train_data.npy")
train_label_path = os.path.join(dataset_path, "train_label.npy")
test_data_path = os.path.join(dataset_path, "test_data.npy")
test_label_path = os.path.join(dataset_path, "test_label.npy")

def calc_KL(p, q):
    return torch.sum(p*(torch.log(p + 1e-7) - torch.log(q + 1e-7)))

def calc_mse(p, q):
    return torch.sum((p - q) ** 2)

def train_attack(net, data, label, test_data, test_label, optimizer):
    train_set  = torch.utils.data.TensorDataset(torch.from_numpy(data), torch.from_numpy(label))
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=128, shuffle=True)
    for epoch in range(n_epochs):
        net.train()
        loss_tot, tot = 0.0, 0
        tf = time.time()
        for input, lbl in train_loader:
            input = input.to(device)
            lbl = lbl.to(device)
            bs = input.size(0)
            out = net(input)
            out = F.softmax(out, dim=1)
            loss = calc_KL(out, lbl) / bs
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_tot += loss.item() * bs
            tot += bs

        test_kl, test_mse = eval_attack(net, test_data, test_label)
        interval = time.time() - tf
        train_loss = loss_tot / tot
    print("Epoch:{}\tTime:{:.2f}\tTrain Loss:{:.4f}\tTrain Acc:{:.2f}\tTrain MSE:{:.2f}".format(
        epoch, interval, train_loss, test_kl, test_mse))


def eval_attack(net, data, label):
    net.eval()
    KL, mse = 0.0, 0.0
    test_set  = torch.utils.data.TensorDataset(torch.from_numpy(data), torch.from_numpy(label))
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=64, shuffle=False)
    for input,lbl in test_loader:
        input = input.to(device)
        lbl = lbl.to(device)
        out = net(input)
        out = F.softmax(out, dim=1)
        KL += calc_KL(out, lbl).item()
        mse += calc_mse(out, lbl).item()

    return KL / data.shape[0], mse / data.shape[0]

if __name__ == "__main__":
    attack_net = models.AttackNet()
    attack_net = nn.DataParallel(attack_net).cuda()
    optimizer = torch.optim.Adam(attack_net.parameters(), lr=lr)
    
    train_data = np.load(train_data_path)
    train_label = np.load(train_label_path)
    test_data = np.load(test_data_path)
    test_label = np.load(test_label_path)
    attack_kl, attack_mse = 0.0, 0.0
    for i in range(10):
        train_attack(attack_net, train_data, train_label, test_data, test_label, optimizer)
        kl, mse = eval_attack(attack_net, test_data, test_label)
        attack_kl += kl
        attack_mse += mse
    print("Attack KL:{:.2f}\tAttack MSE:{:.2f}".format(attack_kl, attack_mse))
    #torch.save({'state_dict':attack_net.state_dict()}, os.path.join(save_path, "AttackNet_single.tar"))

