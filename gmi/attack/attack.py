import torch, os, time, random, generator, discri, classify, utils
import numpy as np 
import torch.nn as nn
import torchvision.utils as tvls

device = "cuda"
num_classes = 1000
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
root_path = "./result/CAttack"
log_path = os.path.join(root_path, "attack_logs")
os.makedirs(log_path, exist_ok=True)

mode = "vib"

train_avg = torch.load('./attack_dataset/Celeba/train_avg.pt')
test_avg = torch.load('./attack_dataset/Celeba/test_avg.pt')

def eval_l2(img, iden):
    num = img.shape[0]
    train_l2, test_l2 = 0.0, 0.0
    for i in range(num):
        lbl = iden[i].item()
        rec = img[i, :, :, :]
        train_l2 += torch.mean((rec - train_avg[lbl, :, :, :]) ** 2).item()
        test_l2 += torch.mean((rec - test_avg[lbl, :, :, :]) ** 2).item()
    return train_l2 / num, test_l2 / num
    

def inversion(G, D, T, E, iden, lr=2e-2, momentum=0.9, lamda=100, iter_times=1500, clip_range=1, verbose=False):
    iden = iden.view(-1).long().cuda()
    criterion = nn.CrossEntropyLoss().cuda()
    bs = iden.shape[0]
    
    G.eval()
    D.eval()
    T.eval()
    E.eval()

    max_score = torch.zeros(bs).float()
    max_iden = torch.zeros(bs)
    z_hat = torch.zeros(bs, 100).float()
    flag = torch.zeros(bs)

    for random_seed in range(5):
        tf = time.time()

        torch.manual_seed(random_seed) 
        torch.cuda.manual_seed(random_seed) 
        np.random.seed(random_seed) 
        random.seed(random_seed)

        z = torch.randn(bs, 100).cuda().float()
        z.requires_grad = True
        v = torch.zeros(bs, 100).cuda().float()

        for i in range(iter_times):
            fake = G(z)
            label = D(fake)
            out = T(fake)[-1]

            if z.grad is not None:
                z.grad.data.zero_()

            Prior_Loss = - label.mean()
            Iden_Loss = criterion(out, iden)
            Total_Loss = Prior_Loss + lamda * Iden_Loss

            Total_Loss.backward()

            v_prev = v.clone()
            gradient = z.grad.data
            v = momentum * v - lr * gradient
            z = z + ( - momentum * v_prev + (1 + momentum) * v)
            z = torch.clamp(z.detach(), -clip_range, clip_range).float()
            z.requires_grad = True

            Prior_Loss_val = Prior_Loss.item()
            Iden_Loss_val = Iden_Loss.item()



            if verbose:
                if (i+1) % 300 == 0:
                    fake_img = G(z.detach())
                    eval_prob = E(utils.low2high(fake_img))[-1]
                    eval_iden = torch.argmax(eval_prob, dim=1).view(-1)
                    acc = iden.eq(eval_iden.long()).sum().item() * 1.0 / bs
                    print("Iteration:{}\tPrior Loss:{:.2f}\tIden Loss:{:.2f}\tAttack Acc:{:.2f}".format(i+1, Prior_Loss_val, Iden_Loss_val, acc))
    
        fake = G(z)
        score = T(fake)[-1]
        eval_prob = E(utils.low2high(fake))[-1]
        eval_iden = torch.argmax(eval_prob, dim=1).view(-1)

        cnt = 0
        for i in range(bs):
            gt = iden[i].item()
            if score[i, gt].item() > max_score[i].item():
                max_score[i] = score[i, gt]
                max_iden[i] = eval_iden[i]
                z_hat[i, :] = z[i, :]
            if eval_iden[i].item() == gt:
                cnt += 1
                flag[i] = 1

        if verbose:
            interval = time.time() - tf
            print("Time:{:.2f}\tAcc:{:.2f}\t".format(interval, cnt * 1.0 / 100))

    final_res = G(z_hat.cuda()).cpu()
    train_l2, test_l2 = eval_l2(final_res, iden)
    
    
    correct = 0
    for i in range(bs):
        gt = iden[i].item()
        if max_iden[i].item() == gt:
            correct += 1

    correct_5 = torch.sum(flag)
    acc, acc_5 = correct * 1.0 / bs, correct_5 * 1.0 / bs
    print("Acc:{:.2f}\tAcc5:{:.2f}\tTrain l2:{:.4f}\tTest l2:{:.4f}".format(acc, acc_5, train_l2, test_l2))
    
    #tvls.save_image(final_res, "../../vib1.png", padding=0, nrow=1)
    return final_res
    

if __name__ == '__main__':
    target_path = os.path.join(root_path, "target_models/VGG16_vib2_80.86.tar")
    target_name = target_path.split('/')[-1].split(".tar")[0]
    log_file = "attack_{}_v200_avg.txt".format(target_name)
    utils.Tee(os.path.join(log_path, log_file), 'w')
    print(target_name)
    if mode == "vib":
        T = classify.VGG16_vib(num_classes)
    else:
        T = classify.VGG16(num_classes)
    T = nn.DataParallel(T).cuda()
    ckp_T = torch.load(target_path)['state_dict']
    utils.load_my_state_dict(T, ckp_T)

    e_path = os.path.join(root_path, "target_models/FaceNet_96.81.tar")
    E = classify.FaceNet(num_classes)
    E = nn.DataParallel(E).cuda()
    ckp_E = torch.load(e_path)['state_dict']
    utils.load_my_state_dict(E, ckp_E)

    g_path = os.path.join(root_path, "attack_result/models_celeba_gan/celeba_G_v200.tar")
    G = generator.Generator()
    G = nn.DataParallel(G).cuda()
    ckp_G = torch.load(g_path)['state_dict']
    utils.load_my_state_dict(G, ckp_G)

    d_path = os.path.join(root_path, "attack_result/models_celeba_gan/celeba_D_v200.tar")
    D = discri.DGWGAN()
    D = nn.DataParallel(D).cuda()
    ckp_D = torch.load(d_path)['state_dict']
    utils.load_my_state_dict(D, ckp_D)
    
    res_list = []

    for j in range(3):
        iden = torch.zeros(100)
        for i in range(100):
            iden[i] = j*100+i
        res = inversion(G, D, T, E, iden, verbose=True)
        res_list.append(res)
        
    
