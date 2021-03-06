import os, gc, sys
import json, PIL, time, random
import torch
import math
import numpy as np
from sklearn.model_selection import train_test_split
from torch.nn.modules.loss import _Loss
import torch.utils.data as data
from torchvision import transforms
import torch.nn.functional as F 

class ImageFolder(data.Dataset):
    def __init__(self, args, file_path, data_type):
        self.img_path = args["dataset"]["img_path"]
        self.img_list = os.listdir(self.img_path)
        self.data_type = data_type
        if args['dataset']['name'] == "celeba":
            self.processor = self.get_celeba_processor()
        else:
            self.processor = self.get_processor()
        
        self.name_list, self.label_list = self.get_list(file_path) 
        self.image_list = self.load_img()
        self.num_img = len(self.image_list)
        self.n_classes = args["dataset"]["n_classes"]
        print("Load " + str(self.num_img) + " images")

    def get_list(self, file_path):
        name_list, label_list = [], []
        f = open(file_path, "r")
        for line in f.readlines():
            img_name, iden = line.strip().split(' ')
            name_list.append(img_name)
            label_list.append(int(iden))

        return name_list, label_list

    
    def load_img(self):
        img_list = []
        for i, img_name in enumerate(self.name_list):
            if img_name.endswith(".png"):
                path = self.img_path + "/" + img_name
                img = PIL.Image.open(path)
                img = img.convert('L')
                img_list.append(img)
        return img_list

    def get_processor(self):
        proc = []
        if self.data_type == "train":
            proc.append(transforms.RandomHorizontalFlip(p=0.5))
            proc.append(transforms.ToTensor())
        else:
            proc.append(transforms.ToTensor())
            
        return transforms.Compose(proc)
    
    def get_celeba_processor(self):
        re_size = 64
        crop_size = 108
        
        offset_height = (218 - crop_size) // 2
        offset_width = (178 - crop_size) // 2
        crop = lambda x: x[:, offset_height:offset_height + crop_size, offset_width:offset_width + crop_size]

        proc = []
        if self.data_type == "train":
            proc.append(transforms.ToTensor())
            proc.append(transforms.Lambda(crop))
            proc.append(transforms.ToPILImage())
            proc.append(transforms.Resize((re_size, re_size)))
            proc.append(transforms.RandomHorizontalFlip(p=0.5))
            proc.append(transforms.ToTensor())
        else:
            proc.append(transforms.ToTensor())
            proc.append(transforms.Lambda(crop))
            proc.append(transforms.ToPILImage())
            proc.append(transforms.Resize((re_size, re_size)))
            proc.append(transforms.ToTensor())
            
        return transforms.Compose(proc)

    def __getitem__(self, index):
        img = self.processor(self.image_list[index])
        label = self.label_list[index]
        return img, label

    def __len__(self):
        return self.num_img


