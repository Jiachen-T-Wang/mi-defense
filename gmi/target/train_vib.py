import torch, os, engine, model, utils, sys
import numpy as np 
import torch.nn as nn
from copy import deepcopy
from sklearn.model_selection import train_test_split

dataset_name = "celeba"
device = "cuda"
root_path = "./result/CAttack"
log_path = os.path.join(root_path, "target_logs")
model_path = os.path.join(root_path, "target_models")
os.makedirs(model_path, exist_ok=True)
os.makedirs(log_path, exist_ok=True)

model_name = "VGG16"
mode = "vib4"
n_epochs = 50
lr = 1e-2
momentum = 0.9
weight_decay = 1e-4
beta = 0.003

os.environ["CUDA_VISIBLE_DEVICES"] = "2"


def main(args, model_name, trainloader, testloader):
    n_classes = args["dataset"]["n_classes"]
    net = model.VGG16_vib(n_classes)

    optimizer = torch.optim.SGD(params=net.parameters(),
                                lr=lr, 
                                momentum=momentum, 
                                weight_decay=weight_decay)
    
    criterion = nn.CrossEntropyLoss().cuda()
    net = torch.nn.DataParallel(net).to(device)
    
    best_model, best_acc = engine.train_vib(args, net, criterion, optimizer, trainloader, testloader, n_epochs, beta)
    
    torch.save({'state_dict':best_model.state_dict()}, os.path.join(model_path, "{}_{:.2f}.tar").format(model_name, best_acc))

if __name__ == '__main__':
    file = "./config/" + dataset_name + ".json"
    args = utils.load_json(json_file=file)
    
    log_file = "{}_{}.txt".format(model_name, mode)
    utils.Tee(os.path.join(log_path, log_file), 'w')

    print(log_file)
    print("---------------------Training [%s]---------------------" % model_name)
    utils.print_params(args["dataset"], args[model_name], dataset=args['dataset']['name'])
    
    train_file = args['dataset']['train_file']
    test_file = args['dataset']['test_file']
    trainloader = utils.init_dataloader(args, train_file, mode="train")
    testloader = utils.init_dataloader(args, test_file, mode="test")
    
    main(args, model_name, trainloader, testloader)
