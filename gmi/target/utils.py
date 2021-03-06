import numpy as np
import torch, random, sys, json, time, dataloader, copy
import torch.nn as nn
from datetime import datetime
from torch.utils.data import sampler
from collections import defaultdict
from torch.autograd import Variable

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

def load_state_dict(self, state_dict):
    own_state = self.state_dict()
    for name, param in state_dict.items():
        if name not in own_state:
            print(name)
            continue
        own_state[name].copy_(param.data)

def load_pretrain(self, state_dict):
    own_state = self.state_dict()
    for name, param in state_dict.items():
        if name.startswith("module.fc_layer"):
            continue
        if name not in own_state:
            print(name)
            continue
        own_state[name].copy_(param.data)

def load_params(self, model):
    own_state = self.state_dict()
    for name, param in model.named_parameters():
        if name not in own_state:
            print(name)
            continue
        own_state[name].copy_(param.data)

def load_json(json_file):
    with open(json_file) as data_file:
        data = json.load(data_file)
    return data

def print_params(info, params, dataset=None):
    print('-----------------------------------------------------------------')
    if dataset is not None:
        print("Dataset: %s" % dataset)
        print("Running time: %s" % datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    for i, (key, value) in enumerate(info.items()):
        if i >=3: 
            print("%s: %s" % (key, str(value)))
    for i, (key, value) in enumerate(params.items()):
        print("%s: %s" % (key, str(value)))
    print('-----------------------------------------------------------------')

def init_dataloader(args, file_path, mode="gan"):
    tf = time.time()
    model_name = args['dataset']['model_name']
    bs = args[model_name]['batch_size']
    if args['dataset']['name'] == "celeba":
        data_set = dataloader.ImageFolder(args, file_path, mode)

    sampler = RandomIdentitySampler(data_set, args[model_name]['batch_size'], args[model_name]['instance'])
        
    if mode == "train":
        data_loader = torch.utils.data.DataLoader(data_set,
                                                  sampler=sampler,
                                                  batch_size=bs,
                                                  num_workers=4,
                                                  pin_memory=True)
    elif mode == "test":
        data_loader = torch.utils.data.DataLoader(data_set,
                                                  shuffle=False,
                                                  batch_size=bs,
                                                  num_workers=4,
                                                  pin_memory=True)
    
    interval = time.time() - tf
    print('Initializing data loader took {:.2f}'.format(interval))
    return data_loader

def weights_init_kaiming(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
        nn.init.constant_(m.weight, 1.0)
        nn.init.constant_(m.bias, 0.0)


def weights_init_classifier(m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)



