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
import torch.nn.functional as F

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

device = "cuda"
dataset_name = "facescrub"
root_path = "./result/align"
save_model_path = os.path.join(root_path, "attack_models")
save_log_path = os.path.join(root_path, "attack_logs")
save_img_path = os.path.join(root_path, "attack_imgs")

target_name = "SimpleCNN"
target_mode = sys.argv[1]
target_path = os.path.join(root_path, "target_models/{}_{}.tar".format(target_name, target_mode))
attack_name = "Inversion"
eval_path = "./result/align/target_models/VGG16_reg.tar"

os.makedirs(save_img_path, exist_ok=True)
os.makedirs(save_log_path, exist_ok=True)
os.makedirs(save_model_path, exist_ok=True)

noise_vector = torch.ones(1, 530).float() / 530
w = 0.0

if target_mode == 'reg':
    w = float(sys.argv[2])

def psnr(img1, img2):
    img1 = img1.detach().cpu().numpy()
    img2 = img2.detach().cpu().numpy()
    mse = numpy.mean((img1 - img2) ** 2 )
    if mse == 0:
        return 100
    PIXEL_MAX = 1.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

def eval_net(attack_net, target_net, dataloader, mode, epoch, eval_model):
    attack_net.eval()
    cnt, l2_tot, hit = 0, 0, 0
    noise = noise_vector.to(device)
    with torch.no_grad():
        for img, lbl in dataloader:
            img = img.to(device)
            lbl = lbl.to(device)
            bs = img.size(0)
            if target_mode.startswith("vib"):
                out_prob = target_net(img)[0].detach().cpu().numpy()
            else:
                out_prob = target_net(img).detach().cpu().numpy()
            in_prob = torch.from_numpy(out_prob).to(device)
            in_prob = (1 - w) * in_prob + w * noise
            rec_img = attack_net(in_prob)
            rec_lbl = torch.argmax(eval_model(rec_img), dim=1)
            
            hit += torch.sum(rec_lbl == lbl).item()
            l2 = torch.sqrt(torch.mean((rec_img - img) ** 2)).item()
            l2_tot += l2 * bs
            cnt += bs

    if (epoch+1) % 10 == 0:
        out_img = torch.cat([img.detach().cpu(), rec_img.detach().cpu()], dim=0)
        utils.save_tensor_images(out_img, 
                                 os.path.join(save_img_path, 'recover_{}_{}.png'.format(mode, epoch)), 
                                 nrow=img.size(0))

    return l2_tot * 1.0 / cnt, hit * 100.0 / cnt

def main(args, trainloader, testloader, eval_model):
    target_net = utils.get_model(args, target_name, target_mode)
    target_net = torch.nn.DataParallel(target_net).to(device)
    utils.load_state_dict(target_net, target_path)
    target_net.eval()
    eval_model.eval()

    attack_net = utils.get_attack_model(args, attack_name)
    attack_net = torch.nn.DataParallel(attack_net).to(device)

    optimizer = torch.optim.Adam(attack_net.parameters(), 
                                lr=args[attack_name]['learning_rate'], 
                                betas=(0.5, 0.999), 
                                amsgrad=True)
    
    best_acc_train, best_l2_train = 0, 1e9
    best_acc_test, best_l2_test = 0, 1e9
    n_epochs = args[attack_name]['epochs']
    noise = noise_vector.to(device)

    # n_epochs = 1
    
    print("Start Training!")
    for e in range(n_epochs):
        tf = time.time()
        attack_net.train()
        for img, __ in trainloader:
            img = img.to(device)
            #utils.save_tensor_images(img, os.path.join(save_img_path, 'sample.png'), nrow=8)
            if target_mode.startswith('vib'):
                out_prob = target_net(img)[0].detach().cpu().numpy()
            else:
                out_prob = target_net(img).detach().cpu().numpy()
            in_prob = torch.from_numpy(out_prob).to(device)
            in_prob = (1 - w) * in_prob + w * noise
            rec_img = attack_net(in_prob)
            loss = F.mse_loss(img, rec_img)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        train_l2, train_acc = eval_net(attack_net, target_net, trainloader, "train", e, eval_model)
        test_l2, test_acc = eval_net(attack_net, target_net, testloader, "test", e, eval_model)
        
        if test_acc > best_acc_test:
            best_acc_test = test_acc
            best_model = deepcopy(attack_net)
        if test_l2 < best_l2_test:
            best_l2_test = test_l2

        if train_acc > best_acc_train:
            best_acc_train = train_acc
        if train_l2 < best_l2_train:
            best_l2_train = train_l2

        interval = time.time() - tf
        print("Epoch:{}\tTime:{:.2f}\tTrain L2:{:.4f}\tTrain Acc:{:.2f}\tTest L2:{:.4f}\tTest Acc:{:.2f}".format(e, interval, train_l2, train_acc, test_l2, test_acc))

    torch.save({'state_dict':best_model.state_dict()}, os.path.join(save_model_path, "attack_{}_{}.tar".format(attack_name, target_mode)))

    print("The best train acc is {:.2f}".format(train_acc))
    print("The best train l2 is {:.3f}".format(train_l2))
    print("The best test acc is {:.2f}".format(test_acc))
    print("The best test l2 is {:.3f}".format(test_l2))
    print("==============================================================================")


if __name__ == '__main__':
    file = dataset_name + ".json" 
    args = utils.load_params(file)
    if w > 0:
        log_file = "attack" + '_' + target_name + '_{}_{}.txt'.format(target_mode, w)
    else:
        log_file = "attack" + '_' + target_name + '_{}.txt'.format(target_mode)
    logger = utils.Tee(os.path.join(save_log_path, log_file), 'w')
    utils.print_params(args)
    
    train_file = args['dataset']['test_file']
    test_file = args['dataset']['train_file']
    trainloader = utils.init_dataloader(args, train_file, mode="train")
    testloader = utils.init_dataloader(args, test_file, mode="test")
    
    eval_model = utils.get_model(args, "VGG16", "reg")
    eval_model = torch.nn.DataParallel(eval_model).to(device)
    utils.load_state_dict(eval_model, eval_path)

    save_img_path = os.path.join(save_img_path, "attack_{}_{}".format(target_name, target_mode))
    os.makedirs(save_img_path, exist_ok=True)
    main(args, trainloader, testloader, eval_model)
