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

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = "cuda"

def eval_net(net, dataloader):
    net.eval()
    cnt, acc = 0, 0
    for img, label in dataloader:
        img, label = img.to(device), label.to(device)
        label = label.view(-1)
        out_prob = net(img)
        out_label = torch.argmax(out_prob, dim=1).view(-1)
        acc += torch.sum(out_label == label).item()
        cnt += img.size(0)

    return acc * 100.0 / cnt

root_path = "./result/align"
model_path = os.path.join(root_path, "result_models/VGG16_reg.tar")
model_name = "VGG16"
dataset_name = "facescrub"

if __name__ == '__main__':
    file = dataset_name + ".json"
    args = utils.load_params(file)
    file_path = args['dataset']['file_path']
    print("Use GPU:{}".format(os.environ["CUDA_VISIBLE_DEVICES"]))
    test_file = os.path.join(file_path, "test_list.txt")
    testloader = utils.init_dataloader(args, test_file, mode="test")
    net = utils.get_model(args, model_name)
    net = torch.nn.DataParallel(net).to(device)
    utils.load_state_dict(net, model_path)
    acc = eval_net(net, testloader)
    print("ACC:{:.2f}".format(acc))

