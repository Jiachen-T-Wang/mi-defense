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

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

lr = 1e-3
n_epochs = 50
n_classes = 10
batch_size = 64
device = "cuda"

root_path = "/opt/tiger/vehiclereid/result/update"
model_v = "vib2-2"
dataset_path = os.path.join(root_path, "single_dataset/{}".format(model_v))


train_data_path = os.path.join(dataset_path, "train_data.npy")
train_label_path = os.path.join(dataset_path, "train_label.npy")
test_data_path = os.path.join(dataset_path, "test_data.npy")
test_label_path = os.path.join(dataset_path, "test_label.npy")

def train_attack(net, data, label, test_data, test_label, optimizer, criterion):
    train_set  = torch.utils.data.TensorDataset(torch.from_numpy(data), torch.from_numpy(label))
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True)
    for epoch in range(n_epochs):
        net.train()
        loss_tot, tot = 0.0, 0
        tf = time.time()
        for input, lbl in train_loader:
            input, lbl = input.to(device), lbl.to(device)
            bs = lbl.shape[0]
            out = net(input)
            loss = criterion(out, lbl)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_tot += loss.item() * bs
            tot += bs

        train_accuracy = eval_attack(net, data, label)
        test_accuracy = eval_attack(net, test_data, test_label)
        interval = time.time() - tf
        train_loss = loss_tot / tot
    
    print("Epoch:{}\tTime:{:.2f}\tTrain Loss:{:.4f}\tTrain Acc:{:.2f}\tTest Acc:{:.2f}".format(
        epoch, interval, train_loss, train_accuracy, test_accuracy))


def eval_attack(net, data, label):
    net.eval()
    correct = 0
    test_set  = torch.utils.data.TensorDataset(torch.from_numpy(data), torch.from_numpy(label))
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=64, shuffle=False)
    for input, lbl in test_loader:
        input, lbl = input.to(device), lbl.to(device)
        out = net(input)
        predict = torch.argmax(out, dim=1)
        correct += predict.eq(lbl).sum().item()

    return correct * 100.0 / data.shape[0]

if __name__ == "__main__":
    attack_net = models.AttackNet()
    attack_net = nn.DataParallel(attack_net).cuda()
    optimizer = torch.optim.Adam(attack_net.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss().cuda()

    train_data = np.load(train_data_path)
    train_label = np.load(train_label_path)
    test_data = np.load(test_data_path)
    test_label = np.load(test_label_path)
    attack_acc = 0.0
    for i in range(10):
        train_attack(attack_net, train_data, train_label, test_data, test_label, optimizer, criterion)
        attack_acc += eval_attack(attack_net, test_data, test_label)
    print("Attack acc:{:.2f}".format(attack_acc / 10))
    #torch.save({'state_dict':attack_net.state_dict()}, os.path.join(save_path, "AttackNet_single.tar"))

