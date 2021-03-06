import time
import os
import torch
import models
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from utils import load_state_dict, load_data

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

np.random.seed(1)

n_epochs = 30
n_classes = 10
update_lr = 1e-2
batch_size = 64
update_size = 10
device = "cuda"

root_path = "/opt/tiger/vehiclereid/result/update"
model_v = "reg"
shadow_path = os.path.join(root_path, "result_models/{}/SimpleCNN_{}_shadow.tar".format(model_v, model_v))
target_path = os.path.join(root_path, "result_models/{}/SimpleCNN_{}_target.tar".format(model_v, model_v))
dataset_path = os.path.join(root_path, "multi_dataset/{}".format(model_v))
os.makedirs(dataset_path, exist_ok=True)

def get_dataset(target_net, update_loader, probe_imgs, criterion, num):
    target_net.eval()
    probe_pre = target_net(probe_imgs)[0].view(1, -1)
    delta_set, label_set = [], []
    
    img_list, label_list = [], []
    for imgs, labels in update_loader:
        img_list.append(imgs)
        label_list.append(labels)
    imgs = torch.cat(img_list, dim=0)
    labels = torch.cat(label_list, dim=0)
    #print(imgs.shape, labels.shape)

    probe_res, label_softmax = [], []
    for i in range(num):
        net = deepcopy(target_net)
        net.zero_grad()
            
        indices = list(np.random.choice(range(num), size=update_size, replace=False))
        img, label = imgs[indices, ...].to(device), labels[indices].to(device)
        out = net(img)[0]
        loss = criterion(out, label)
        loss.backward()

        for p in net.parameters():
            p.data.sub_(update_lr*p.grad.data)

        net.eval()
        probe_out = net(probe_imgs)[0].view(1, -1)
        probe_res.append(probe_out.detach())

        one_hot = torch.zeros(1, n_classes)
        for i in range(10):
            one_hot[0, labels[i]] += 1
        one_hot = F.softmax(one_hot, dim=1)
        label_softmax.append(one_hot.detach())

    probe_res = torch.cat(probe_res, dim=0)
    delta = probe_pre - probe_res
    label_softmax = torch.cat(label_softmax, dim=0)
    #print(delta.shape, label_softmax.shape)

    return delta, label_softmax

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

    if model_v.startswith("dp"):
        train_num = 2000
    else:
        train_num = 1000
    train_in, train_label = get_dataset(shadow_net, D_s_u_loader, probe_imgs, criterion, num=train_num)
    test_in, test_label = get_dataset(target_net, D_u_loader, probe_imgs, criterion, num=1000)

    train_data = train_in.detach().cpu().numpy()
    train_label = train_label.detach().cpu().numpy()

    print(train_data.shape, train_label.shape)

    np.save(os.path.join(dataset_path, "train_data.npy"), train_data)
    np.save(os.path.join(dataset_path, "train_label.npy"), train_label)

    test_data = test_in.detach().cpu().numpy()
    test_label = test_label.detach().cpu().numpy()

    print(test_data.shape, test_label.shape)

    np.save(os.path.join(dataset_path, "test_data.npy"), test_data)
    np.save(os.path.join(dataset_path, "test_label.npy"), test_label)

    #train_attack(attack_net, shadow_net, D_s_u_loader, probe_imgs, optimizer, criterion)
    #eval_attack(attack_model, target_net, D_u_loader, probe_imgs, criterion)

    #torch.save({'state_dict':attack_net.state_dict()}, os.path.join(save_path, "AttackNet_single.tar"))

