import time
import os
import torch
import models
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from utils import load_state_dict, load_data

n_epochs = 30
n_classes = 10
update_lr = 1e-2
attack_lr = 1e-3
batch_size = 64
device = "cuda"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

root_path = "/opt/tiger/vehiclereid/result/update"
model_v = "vib2-2"
shadow_path = os.path.join(root_path, "result_models/{}/SimpleCNN_{}_shadow.tar".format(model_v, model_v))
target_path = os.path.join(root_path, "result_models/{}/SimpleCNN_{}_target.tar".format(model_v, model_v))
dataset_path = os.path.join(root_path, "single_dataset/{}".format(model_v))
os.makedirs(dataset_path, exist_ok=True)

noise_vector = torch.Tensor([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]).view(1, -1).to(device)
w = 0.0

def get_dataset(target_net, update_loader, probe_imgs, criterion):
    target_net.eval()
    probe_pre = target_net(probe_imgs)[0]
    probe_pre = probe_pre * (1 - w) + w * noise_vector
    probe_pre = probe_pre.view(1, -1)
    delta_set, label_set = [], []

    for imgs, labels in update_loader:
        bs = imgs.size(0)
        imgs, labels = imgs.to(device), labels.to(device)
        probe_res = []
        for i in range(bs):
            net = deepcopy(target_net)
            net.zero_grad()
            img, label = imgs[i, :, :, :].unsqueeze(0), labels[i].unsqueeze(0)
            out = net(img)[0]
            
            loss = criterion(out, label)
            loss.backward()

            for p in net.parameters():
                p.data.sub_(update_lr*p.grad.data)
            
            net.eval()
            probe_out = net(probe_imgs)[0]
            probe_out = probe_out * (1 - w) + w * noise_vector
            probe_out = probe_out.view(1, -1)
            probe_res.append(probe_out.detach())

        probe_res = torch.cat(probe_res, dim=0)
        delta = probe_pre - probe_res
        delta_set.append(delta.detach())
        label_set.append(labels.detach())

    return delta_set, label_set

def train_attack(net, data, label, test_data, test_label, optimizer, criterion, verbose=False):
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
    
    if verbose:
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

__, __, D_s_u_loader, D_u_loader, probe_imgs, probe_labels, __ = load_data(batch_size=batch_size)
probe_imgs, probe_labels = probe_imgs.to(device), probe_labels.to(device)

def work(target_net, shadow_net):
    criterion = nn.CrossEntropyLoss().cuda()

    train_in, train_label = get_dataset(shadow_net, D_s_u_loader, probe_imgs, criterion)
    test_in, test_label = get_dataset(target_net, D_u_loader, probe_imgs, criterion)

    train_data = torch.cat(train_in, dim=0)
    train_data = train_data.cpu().numpy()
    train_label = torch.cat(train_label, dim=0)
    train_label = train_label.cpu().numpy()

    #print(train_data.shape, train_label.shape)

    np.save(os.path.join(dataset_path, "train_data.npy"), train_data)
    np.save(os.path.join(dataset_path, "train_label.npy"), train_label)

    test_data = torch.cat(test_in, dim=0)
    test_data = test_data.cpu().numpy()
    test_label = torch.cat(test_label, dim=0)
    test_label = test_label.cpu().numpy()

    #print(test_data.shape, test_label.shape)

    np.save(os.path.join(dataset_path, "test_data.npy"), test_data)
    np.save(os.path.join(dataset_path, "test_label.npy"), test_label)
    
    attack_net = models.AttackNet()
    attack_net = nn.DataParallel(attack_net).cuda()
    optimizer = torch.optim.Adam(attack_net.parameters(), lr=attack_lr)
    criterion = nn.CrossEntropyLoss().cuda()

    attack_acc = 0.0
    for i in range(10):
        train_attack(attack_net, train_data, train_label, test_data, test_label, optimizer, criterion)
        attack_acc += eval_attack(attack_net, test_data, test_label)
    attack_acc /= 10
    print("Attack acc:{:.2f}".format(attack_acc))
    return attack_acc

if __name__ == "__main__":
    if model_v.startswith("vib"):
        shadow_net = models.SimpleCNN_VIB(n_classes)
    else:
        shadow_net = models.SimpleCNN(n_classes)
    shadow_net = nn.DataParallel(shadow_net).cuda()
    shadow_ckp = torch.load(shadow_path)['state_dict']
    load_state_dict(shadow_net, shadow_ckp)

    if model_v.startswith("vib"):
        target_net = models.SimpleCNN_VIB(n_classes)
    else:
        target_net = models.SimpleCNN(n_classes)
    target_net = nn.DataParallel(target_net).cuda()
    target_ckp = torch.load(target_path)['state_dict']
    load_state_dict(target_net, target_ckp)
    criterion = nn.CrossEntropyLoss().cuda()

    __, __, D_s_u_loader, D_u_loader, probe_imgs, probe_labels, __ = load_data(batch_size=batch_size)
    probe_imgs, probe_labels = probe_imgs.to(device), probe_labels.to(device)

    train_in, train_label = get_dataset(shadow_net, D_s_u_loader, probe_imgs, criterion)
    test_in, test_label = get_dataset(target_net, D_u_loader, probe_imgs, criterion)

    train_data = torch.cat(train_in, dim=0)
    train_data = train_data.cpu().numpy()
    train_label = torch.cat(train_label, dim=0)
    train_label = train_label.cpu().numpy()

    print(train_data.shape, train_label.shape)

    np.save(os.path.join(dataset_path, "train_data.npy"), train_data)
    np.save(os.path.join(dataset_path, "train_label.npy"), train_label)

    test_data = torch.cat(test_in, dim=0)
    test_data = test_data.cpu().numpy()
    test_label = torch.cat(test_label, dim=0)
    test_label = test_label.cpu().numpy()

    print(test_data.shape, test_label.shape)

    np.save(os.path.join(dataset_path, "test_data.npy"), test_data)
    np.save(os.path.join(dataset_path, "test_label.npy"), test_label)
    
    attack_net = models.AttackNet()
    attack_net = nn.DataParallel(attack_net).cuda()
    optimizer = torch.optim.Adam(attack_net.parameters(), lr=attack_lr)
    criterion = nn.CrossEntropyLoss().cuda()

    attack_acc = 0.0
    for i in range(10):
        train_attack(attack_net, train_data, train_label, test_data, test_label, optimizer, criterion, verbose=True)
        attack_acc += eval_attack(attack_net, test_data, test_label)
    print("Attack acc:{:.2f}".format(attack_acc / 10))

    #torch.save({'state_dict':attack_net.state_dict()}, os.path.join(save_path, "AttackNet_single.tar"))

