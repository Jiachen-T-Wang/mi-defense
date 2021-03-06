import os
import sys
import torch
import models
import torchvision
import numpy as np 
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils as tvls
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler


device = "cuda"
root_path = "/opt/tiger/vehiclereid/result/update"
cifar10_path = os.path.join(root_path, "dataset/CIFAR10")
cifar100_path = os.path.join(root_path, "dataset/CIFAR100")
update_bs = 64
n_classes = 10

sample_imgdir = "./sample_images"
os.makedirs(sample_imgdir, exist_ok=True)

class Tee(object):
    def __init__(self, name, mode):
        self.file = open(name, mode)
        self.stdout = sys.stdout
        sys.stdout = self
    def __del__(self):
        sys.stdout = self.stdout
        self.file.close()
    def write(self, data):
        if not '...' in data:
            self.file.write(data)
        self.stdout.write(data)
        self.flush()
    def flush(self):
        self.file.flush()

def save_tensor_images(images, filename, nrow = None, normalize = True):
    if not nrow:
        tvls.save_image(images, filename, normalize = normalize)
    else:
        tvls.save_image(images, filename, normalize = normalize, nrow = nrow)

def weights_init(m): 
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear): 
        nn.init.xavier_normal_(m.weight.data)
        nn.init.constant_(m.bias, 0)

def label_to_onehot(labels, num_classes=10): 
    one_hot = torch.eye(num_classes)
    return one_hot[labels]

def get_model():
    return models.VGG16(n_classes)
  
def load_state_dict(self, state_dict):
    own_state = self.state_dict()
    for name, param in state_dict.items():
        if name not in own_state:
            print(name)
            continue
        own_state[name].copy_(param.data)  

def load_data(dataset="cifar10", num_sample=1, batch_size=128):
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        #torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    # load training set 
    if dataset.lower() == "cifar10":
        trainset = torchvision.datasets.CIFAR10(cifar10_path, train=True, transform=transform, download=True)
        testset = torchvision.datasets.CIFAR10(cifar10_path, train=False, transform=transform, download=True)
    else:
        trainset = torchvision.datasets.CIFAR100(cifar100_path, train=True, transform=transform, download=True)
        testset = torchvision.datasets.CIFAR100(cifar100_path, train=False, transform=transform, download=True)

    total_size = len(trainset)
    indices = list(range(total_size))
    D_target_idx = indices[:10000]
    D_shadow_train_idx = indices[10000:20000]
    D_shadow_update_idx = indices[20000:30000]
    D_update_idx = indices[30000:31000]
    D_probe_idx = indices[40000:]

    D_t_sampler = SubsetRandomSampler(D_target_idx)
    D_s_t_sampler = SubsetRandomSampler(D_shadow_train_idx)
    D_s_u_sampler = SubsetRandomSampler(D_shadow_update_idx)
    D_u_sampler = SubsetRandomSampler(D_update_idx)
    D_p_sampler = SubsetRandomSampler(D_probe_idx)

    D_t_loader = DataLoader(trainset, batch_size=batch_size, sampler=D_t_sampler)
    D_s_t_loader = DataLoader(trainset, batch_size=batch_size, sampler=D_s_t_sampler)
    D_s_u_loader = DataLoader(trainset, batch_size=update_bs*num_sample, sampler=D_s_u_sampler)
    D_u_loader = DataLoader(trainset, batch_size=update_bs*num_sample, sampler=D_u_sampler)
    D_p_loader = DataLoader(trainset, batch_size=batch_size, sampler=D_p_sampler)
    test_loader = DataLoader(testset, batch_size=batch_size)
    
    statis = [0] * 10
    probe_imgs, probe_labels = [], []
    for imgs, labels in D_p_loader:
        bs = labels.size(0)
        for i in range(bs):
            label = labels[i].item()
            if statis[label] < 10:
                statis[label] += 1
                probe_imgs.append(imgs[i, :, :, :].unsqueeze(0))
                probe_labels.append(labels[i].unsqueeze(0))

    probe_imgs = torch.cat(probe_imgs, dim=0)
    probe_labels = torch.cat(probe_labels, dim=0)

    #save_tensor_images(probe_imgs, os.path.join(sample_imgdir, "sample.png"), nrow=8)
    
    return D_t_loader, D_s_t_loader, D_s_u_loader, D_u_loader, probe_imgs, probe_labels, test_loader
