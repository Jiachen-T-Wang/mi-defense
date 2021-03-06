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
from torch.nn import functional as F


device = "cuda"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
dataset_name = "facescrub"
root_path = "./result/align"
save_model_path = os.path.join(root_path, "target_models")
save_log_path = os.path.join(root_path, "target_logs")
save_img_path = os.path.join(root_path, "target_imgs")
os.makedirs(save_img_path, exist_ok=True)
os.makedirs(save_log_path, exist_ok=True)
os.makedirs(save_model_path, exist_ok=True)

target_name = "SimpleCNN"
target_mode = sys.argv[1]
target_path = os.path.join(root_path, "target_models/{}_{}.tar".format(target_name, target_mode))
n_bins = 100

noise_vector = torch.ones(1, 530).float() / 530
if target_mode=='reg':
    w = float(sys.argv[2])
else:
    w = 0.0


def eval_ece(net, dataloader, n_bins=10):
    net.eval()
    bin_boundaries = torch.linspace(0, 1, n_bins+1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    acc_lst, conf_lst = [], []

    noise = noise_vector.to(device)

    with torch.no_grad():
        for img, label in dataloader:
            img, label = img.to(device), label.to(device)
            label = label.view(-1)
            if target_mode.startswith("vib"):
                out_prob, _, _ = net(img)
            else:
                out_prob = net(img)

            out_prob = (1 - w) * out_prob + w * noise

            prediction = torch.argmax(out_prob, dim=1).view(-1)

            softmax = F.softmax(out_prob, dim=1)

            confidence, _ = torch.max(softmax, 1)

            accuracy = prediction.eq(label)
            acc_lst = acc_lst+accuracy.cpu().tolist()
            conf_lst = conf_lst+confidence.cpu().tolist()


    acc_lst, conf_lst = torch.FloatTensor(acc_lst), torch.FloatTensor(conf_lst)
    ece = torch.zeros(1)
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Calculated |confidence - accuracy| in each bin
        in_bin = conf_lst.gt(bin_lower.item()) * conf_lst.le(bin_upper.item())
        prop_in_bin = in_bin.float().mean()
        if prop_in_bin.item() > 0:
            accuracy_in_bin = acc_lst[in_bin].float().mean()
            avg_confidence_in_bin = conf_lst[in_bin].mean()
            ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

    return ece


def main(args, trainloader, testloader):
    net = utils.get_model(args, target_name, target_mode)
    net = torch.nn.DataParallel(net).to(device)

    utils.load_state_dict(net, target_path)
    net.eval()

    ece_train = eval_ece(net, trainloader, n_bins=n_bins)
    ece_test = eval_ece(net, testloader, n_bins=n_bins)

    print("The Expected Calibration Accuracy on Training Set is %.03f." % (ece_train))
    print("The Expected Calibration Accuracy on Test Set is %.03f." % (ece_test))
    print("==============================================================================")


if __name__ == '__main__':
    file = dataset_name + ".json"
    args = utils.load_params(file)
    train_file = args['dataset']['train_file']
    test_file = args['dataset']['test_file']
    trainloader = utils.init_dataloader(args, train_file, mode="train")
    testloader = utils.init_dataloader(args, test_file, mode="test")

    print(n_bins)
    main(args, trainloader, testloader)
