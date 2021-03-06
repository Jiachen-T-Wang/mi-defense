import os
import torch
import models 
import numpy as np 
import torch.nn as nn
from copy import deepcopy
from utils import load_data, Tee
from get_single_data import work
from dp_training import train_dp
import torch.nn.functional as F

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
lr = 3e-2
momentum = 0.9
noise_multiplier = 0.2
n_epochs = 200
n_classes = 10
batch_size = 128

# [CIFAR10, MNIST]
model_v = "dp4-3"
dataset = "CIFAR10"
root_path = "/opt/tiger/vehiclereid/result/update"
save_path = os.path.join(root_path, "result_models/{}".format(model_v))
log_path = os.path.join(root_path, "logs/{}".format(model_v))
os.makedirs(save_path, exist_ok=True)
os.makedirs(log_path, exist_ok=True)

log_file = "train_scnn_{}.txt".format(model_v)
logger = Tee(os.path.join(log_path, log_file), 'w')

def eval_ece(net, dataloader, n_bins=10):
    net.eval()
    bin_boundaries = torch.linspace(0, 1, n_bins+1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    acc_lst, conf_lst = [], []

    #noise = noise_vector.to(device)

    with torch.no_grad():
        for img, label in dataloader:
            img, label = img.cuda(), label.cuda()
            label = label.view(-1)
            out_prob = net(img)[0]
            
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

    return ece.item()

if __name__ == "__main__":
    D_t_loader, D_s_t_loader, D_s_u_loader, D_u_loader, probe_imgs, probe_labels, test_loader = load_data(dataset, batch_size=batch_size)
    if dataset == "CIFAR10":
        net_shadow = models.SimpleCNN(n_classes)
        net_target = models.SimpleCNN(n_classes)
    else:
        net_shadow = models.MCNN(n_classes)
        net_target = models.MCNN(n_classes)
        
    opt_shadow = torch.optim.SGD(net_shadow.parameters(), lr=lr, momentum=momentum)
    opt_target = torch.optim.SGD(net_target.parameters(), lr=lr, momentum=momentum)
    
    net_shadow = nn.DataParallel(net_shadow).cuda()
    net_target = nn.DataParallel(net_target).cuda()
    loss = nn.CrossEntropyLoss(reduction='none').cuda()

    '''
    best_model, acc, eps = train_dp(net_target, D_t_loader, test_loader, opt_target, loss, n_epochs, batch_size, n_classes, noise_multiplier)
    torch.save({'state_dict':best_model.state_dict()}, os.path.join(save_path, "SimpleCNN_{}_target.tar".format(model_v)))

    best_model, acc, eps = train_dp(net_shadow, D_s_t_loader, test_loader, opt_shadow, loss, n_epochs, batch_size, n_classes, noise_multiplier)
    torch.save({'state_dict':best_model.state_dict()}, os.path.join(save_path, "SimpleCNN_{}_shadow.tar".format(model_v)))
    '''
    print("DP Version:{}".format(model_v))
    print("Learning Rate:{:.2f}".format(lr))
    print("Noise Multiplier:{:.2f}".format(noise_multiplier))
    print("Batch Size:{}".format(batch_size))
    
   
    for i in range(n_epochs):
        target_acc = train_dp(net_target, D_t_loader, test_loader, opt_target, loss, i, batch_size, noise_multiplier=noise_multiplier)
        #shadow_acc = train_dp(net_shadow, D_s_t_loader, test_loader, opt_shadow, loss, i, batch_size, noise_multiplier=noise_multiplier)
        '''
        if n_epochs - i <= 100:
            print("Target Acc:{:.2f}".format(target_acc))
            attack_acc = work(net_target, net_shadow)
        '''
        train_ece, test_ece = eval_ece(net_target, D_t_loader), eval_ece(net_target, test_loader)
        print("Train ECE:{:.2f}\tTest ECE:{:.2f}".format(train_ece, test_ece))


