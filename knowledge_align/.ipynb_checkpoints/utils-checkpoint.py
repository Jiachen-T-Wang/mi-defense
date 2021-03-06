import sys
import time
import json
import model
import torch
import random
import dataloader
import numpy as np
import torch.nn as nn
from datetime import datetime
import torchvision.utils as tvls
from torch.utils.data import sampler

root_path = "/opt/tiger/vehiclereid/result/align"

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

def init_dataloader(args, file_path, mode):
    flag = (mode == "train")
    tf = time.time()
    data_set = dataloader.ImageFolder(args, file_path=file_path, data_type=mode)
    data_loader = torch.utils.data.DataLoader(data_set,
                                              batch_size=args['dataset']['batch_size'],
                                              shuffle=flag,
                                              num_workers=2,
                                              pin_memory=True)
        
    interval = time.time() - tf
    print('Initializing data loader took %ds' % interval)
    return data_loader

def load_params(json_file):
    with open(json_file) as data_file:
        data = json.load(data_file)
    return data

def print_params(args):
    print('-----------------------------------------------------------------')
    model_name = args['dataset']['model_name']
    print("Running time: %s" % datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    for i, (key, value) in enumerate(args['dataset'].items()):
        print("%s: %s" % (key, str(value)))
    for i, (key, value) in enumerate(args[model_name].items()):
        print("%s: %s" % (key, str(value)))
    print('-----------------------------------------------------------------')

def save_tensor_images(images, filename, nrow=None, normalize=True, padding=0):
    if not nrow:
        tvls.save_image(images, filename, normalize=normalize, padding=0)
    else:
        tvls.save_image(images, filename, normalize=normalize, nrow=nrow, padding=0)

def get_model(args, model_name, mode=None):
    if model_name.startswith('SimpleCNN'):
        if mode.startswith("vib"):
            return model.Classifier_VIB(args[model_name]['nc'],
                                args[model_name]['ndf'],
                                args['dataset']['n_classes'])
        else:
            return model.Classifier(args[model_name]['nc'],
                                args[model_name]['ndf'],
                                args['dataset']['n_classes'])
    elif model_name.startswith('VGG16'):
        return model.VGG16(args['dataset']['n_classes'])
    else:
        print("Model Name Error")
        exit()

def get_attack_model(args, model_name):
    if model_name.startswith('Inversion'):
        return model.Inversion(args[model_name]['nc'],
                               args[model_name]['ndf'],
                               args['dataset']['n_classes'])
    else:
        print("Model Name Error")
        exit()

def load_state_dict(self, model_path):
    ckp = torch.load(model_path)['state_dict']
    own_state = self.state_dict()
    for name, param in ckp.items():
        if name not in own_state:
            print(name)
            continue
        own_state[name].copy_(param.data)