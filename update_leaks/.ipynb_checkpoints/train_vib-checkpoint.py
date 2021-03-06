import os
import torch
import models 
import random
import numpy as np 
import torch.nn as nn
from engine import train_vib
from copy import deepcopy
from utils import load_data, Tee
from get_single_data import work

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

lr = 5e-3
n_epochs = 200
n_classes = 10
batch_size = 128
beta = 0.01
momentum = 0.9
model_v = "vib4-1"

root_path = "/opt/tiger/vehiclereid/result/update"
save_path = os.path.join(root_path, "result_models/{}".format(model_v))
log_path = os.path.join(root_path, "logs/{}".format(model_v))
os.makedirs(save_path, exist_ok=True)
os.makedirs(log_path, exist_ok=True)
log_file = "train_scnn_{}.txt".format(model_v)
logger = Tee(os.path.join(log_path, log_file), 'w')

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
def eval_ece(net, data_loader):
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
            out_prob, _, _ = net(img)
            
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

if __name__ == "__main__":
    seed = 0
    set_random_seed(seed)
    
    D_t_loader, D_s_t_loader, D_s_u_loader, D_u_loader, probe_imgs, probe_labels, test_loader = load_data(batch_size=batch_size)

    net_shadow, net_target = models.SimpleCNN_VIB(n_classes), models.SimpleCNN_VIB(n_classes)
    opt_shadow = torch.optim.SGD(net_shadow.parameters(), lr=lr, momentum=momentum)
    opt_target = torch.optim.SGD(net_target.parameters(), lr=lr, momentum=momentum)
    net_target = nn.DataParallel(net_target).cuda()
    net_shadow = nn.DataParallel(net_shadow).cuda()
    loss = nn.CrossEntropyLoss().cuda()
    
    for i in range(n_epochs):
        target_acc = train_vib(net_target, D_t_loader, test_loader, opt_target, loss, i, beta=beta)
        shadow_acc = train_vib(net_shadow, D_s_t_loader, test_loader, opt_shadow, loss, i, beta=beta)
        train_ece, test_ece = eval_ece(net_target, D_t_loader), eval_ece(net_target, test_loader)
        print("Train ECE:{:.2f}\tTest ECE:{:.2f}".format(train_ece, test_ece))
        '''
        if n_epochs - i <= 170:
            print("Target Acc:{:.2f}".format(target_acc))
            attack_acc = work(net_target, net_shadow)
        '''
    
    
    #torch.save({'state_dict':best_model.state_dict()}, os.path.join(save_path, "SimpleCNN_{}_target.tar".format(model_v)))
    #torch.save({'state_dict':best_model.state_dict()}, os.path.join(save_path, "SimpleCNN_{}_shadow.tar".format(model_v)))
