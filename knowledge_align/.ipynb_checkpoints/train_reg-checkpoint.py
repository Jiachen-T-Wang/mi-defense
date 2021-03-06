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

device = "cuda"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
dataset_name = "facescrub"
root_path = "/opt/tiger/vehiclereid/result/align"
save_model_path = os.path.join(root_path, "target_models")
save_log_path = os.path.join(root_path, "target_logs")
save_img_path = os.path.join(root_path, "target_imgs")
os.makedirs(save_img_path, exist_ok=True)
os.makedirs(save_log_path, exist_ok=True)
os.makedirs(save_model_path, exist_ok=True)

model_name = "VGG16"
mode = "reg"

def eval_net(net, dataloader, criterion):
    net.eval()
    cnt, acc, loss_tot = 0, 0, 0
    for img, label in dataloader:
        img, label = img.to(device), label.to(device)
        label = label.view(-1)
        out_prob = net(img)
        loss = criterion(out_prob, label)
        out_label = torch.argmax(out_prob, dim=1).view(-1)
        acc += torch.sum(out_label == label).item()
        cnt += img.size(0)
        loss_tot += loss.item() * label.size(0)

    return loss_tot * 1.0 / cnt, acc * 100.0 / cnt

def main(args, trainloader, testloader):
    net = utils.get_model(args, model_name)
    net = torch.nn.DataParallel(net).to(device)
    
    optimizer = torch.optim.SGD(net.parameters(), 
                                lr=args[model_name]['learning_rate'], 
                                momentum=args[model_name]['momentum'], 
                                weight_decay=args[model_name]['weight_decay'])
    criterion = nn.CrossEntropyLoss().cuda()
    best_ACC = 0
    n_epochs = args[model_name]['epochs']
    
    print("Start Training!")
    for e in range(n_epochs):
        tf = time.time()
        net.train()
        for img, label in trainloader:
            img, label = img.to(device), label.to(device)
            
            utils.save_tensor_images(img, os.path.join(save_img_path, 'sample.png'), nrow=8)
            label = label.view(-1)
            out_prob = net(img)
            loss = criterion(out_prob, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        train_loss, train_acc = eval_net(net, trainloader, criterion)
        test_loss, test_acc = eval_net(net, testloader, criterion)
        if test_acc > best_ACC:
            best_ACC = test_acc
            best_model = deepcopy(net)

        interval = time.time() - tf
        print("Epoch:{}\tTime:{:.0f}\tTrain Loss:{:.2f}\tTrain Acc:{:.2f}\tTest Acc:{:.2f}".format(e, interval, train_loss, train_acc, test_acc))

    torch.save({'state_dict':best_model.state_dict()}, os.path.join(save_model_path, "{}_{}.tar".format(model_name, mode)))

    print("The best accuracy is %.03f." % (best_ACC))
    print("==============================================================================")

if __name__ == '__main__':
    file = dataset_name + ".json"
    args = utils.load_params(file)
    log_file = "train_" + model_name + '_{}.txt'.format(mode)
    logger = utils.Tee(os.path.join(save_log_path, log_file), 'w')

    train_file = args['dataset']['train_file']
    test_file = args['dataset']['test_file']
    trainloader = utils.init_dataloader(args, train_file, mode="train")
    testloader = utils.init_dataloader(args, test_file, mode="test")
    main(args, trainloader, testloader)