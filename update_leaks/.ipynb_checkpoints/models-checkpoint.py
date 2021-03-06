import torch
import numpy as np
import torch.nn as nn
import torchvision.models
import torch.nn.functional as F

class VGG16(nn.Module):
    def __init__(self, n_classes=10):
        super(VGG16, self).__init__()
        model = torchvision.models.vgg16(pretrained=True)
        self.feature = nn.Sequential(*list(model.children())[:-1])
        self.feat_dim = 512 
        self.n_classes = n_classes
        self.classifier = nn.Linear(self.feat_dim, self.n_classes)
        
    def forward(self, x):
        feature = self.feature(x)
        feature = feature.view(feature.size(0), -1)
        res = self.classifier(feature)

        return [res]

class SimpleCNN(nn.Module):
    def __init__(self, n_classes=10):
        super(SimpleCNN, self).__init__()
        self.feature = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(2, 2))
        self.feat_dim = 128 * 4 * 4
        self.n_classes = n_classes
        self.classifier = nn.Linear(self.feat_dim, n_classes)
        
    def forward(self, x):
        feature = self.feature(x)
        feature = feature.view(feature.size(0), -1)
        res = self.classifier(feature)

        return [res]

class SimpleCNN_VIB(nn.Module):
    def __init__(self, n_classes=10):
        super(SimpleCNN_VIB, self).__init__()
        self.feature = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(2, 2))
        self.feat_dim = 128 * 4 * 4
        self.n_classes = n_classes
        self.k = self.feat_dim // 2
        self.st_layer = nn.Linear(self.feat_dim, self.k * 2)
        self.classifier = nn.Linear(self.k, n_classes)
        
    def forward(self, x):
        feature = self.feature(x)
        feature = feature.view(feature.size(0), -1)
        statis = self.st_layer(feature)
        mu, std = statis[:, :self.k], statis[:, self.k:]
        
        std = F.softplus(std-5, beta=1)
        eps = torch.FloatTensor(std.size()).normal_().cuda()
        res = mu + std * eps
        out = self.classifier(res)
        
        return [out, mu, std]

class AlexNet(nn.Module):
    def __init__(self, n_classes=10):
        super(AlexNet, self).__init__()
        model = torchvision.models.alexnet(pretrained=True)
        self.feature = nn.Sequential(*list(model.children())[:-1])
        self.feat_dim = 256 
        self.n_classes = n_classes
        self.classifier = nn.Linear(self.feat_dim, self.n_classes)
        
    def forward(self, x):
        feature = self.feature(x)
        feature = feature.view(feature.size(0), -1)
        print(feature.size())
        res = self.classifier(feature)

        return [res]

class VGG16_VIB(nn.Module):
    def __init__(self, n_classes=10):
        super(VGG16_VIB, self).__init__()
        model = torchvision.models.vgg16(pretrained=True)
        self.feature = nn.Sequential(*list(model.children())[:-1])
        self.feat_dim = 512 
        self.n_classes = n_classes
        self.k = self.feat_dim // 2
        self.st_layer = nn.Linear(self.feat_dim, self.k * 2)
        self.fc_layer = nn.Linear(self.k, self.n_classes)
        
    def forward(self, x):
        feature = self.feature(x)
        feature = feature.view(feature.size(0), -1)
        statis = self.st_layer(feature)
        mu, std = statis[:, :self.k], statis[:, self.k:]
        
        std = F.softplus(std-5, beta=1)
        eps = torch.FloatTensor(std.size()).normal_().cuda()
        res = mu + std * eps
        res = self.fc_layer(res)

        return [res, mu, std]

class AttackNet(nn.Module):
    def __init__(self, input_dim=1000):
        super(AttackNet, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LeakyReLU(0.2))
        self.fc2 = nn.Sequential(
            nn.Linear(128, 64),
            nn.LeakyReLU(0.2))
        self.dropout = nn.Dropout()
        self.fc3 = nn.Linear(64, 10)
            
    def forward(self, x):
        res = self.fc2(self.fc1(x))
        res = self.dropout(res)
        res = self.fc3(res)

        return res


