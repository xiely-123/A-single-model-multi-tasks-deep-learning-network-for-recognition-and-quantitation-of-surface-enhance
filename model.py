# -*- coding: utf-8 -*-
"""
Created on Tue Dec  1 09:12:28 2020

@author: ALW
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class oneConv(nn.Module):
    # 卷积+ReLU函数
    def __init__(self, in_channels, out_channels, kernel_sizes, paddings, dilations):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size = kernel_sizes, padding = paddings, dilation = dilations ),
            #nn.BatchNorm1d(out_channels),
            #nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class ResBlock(nn.Module):
    # 卷积+ReLU函数
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, padding = 1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding = 1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.skip = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=1, padding = 0),
        )
        self.BN_Relu = nn.Sequential(
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),

        )

    def forward(self, x):
        x = self.conv(x) + self.skip(x)
        x = self.BN_Relu(x)
        return x








class ResNet(nn.Module):
    def __init__(self, n_channels, n_classes, n_classes_1, bilinear=False):
        super(ResNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.M_layer1 = oneConv(n_channels,64,3,1,1)
        self.pooling1 = nn.MaxPool1d(kernel_size = 3, stride=2,padding = 1)
        self.M_layer2 = ResBlock(64,128)
        self.pooling2 = nn.MaxPool1d(kernel_size = 5, stride=3,padding = 2)
        self.M_layer3 = ResBlock(128,256)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        #self.fc_1 = nn.Linear(256, n_classes)
        #self.fc_2 = nn.Linear(256, n_classes_1)
        self.fc_1 = torch.nn.Sequential(
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(128, n_classes),
        )
        self.fc_2 = torch.nn.Sequential(
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(128, n_classes_1),
        )
    def forward(self, x):
    
         x = self.pooling1(self.M_layer1(x))
         #print(x.size())
         x = self.pooling2(self.M_layer2(x))
         #print(x.size())
         x = self.avgpool(self.M_layer3(x))
         #print(x.size())         
         out = x.view(-1, 256)
         out_1 = self.fc_1(out)
         out_2 = self.fc_2(out)
         return out_1, out_2




class ResNet_reg(nn.Module):
    def __init__(self, n_channels, n_classes, n_classes_1, bilinear=False):
        super(ResNet_reg, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.M_layer1 = oneConv(n_channels,64,3,1,1)
        self.pooling1 = nn.MaxPool1d(kernel_size = 3, stride=2,padding = 1)
        self.M_layer2 = ResBlock(64,128)
        self.pooling2 = nn.MaxPool1d(kernel_size = 5, stride=3,padding = 2)
        self.M_layer3 = ResBlock(128,256)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        #self.fc_1 = nn.Linear(256, n_classes)
        #self.fc_2 = nn.Linear(256, n_classes_1)
        self.fc_1 = torch.nn.Sequential(
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(128, 1),
        )
        self.fc_2 = torch.nn.Sequential(
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(128, n_classes_1),
        )
    def forward(self, x):
    
         x = self.pooling1(self.M_layer1(x))
         #print(x.size())
         x = self.pooling2(self.M_layer2(x))
         #print(x.size())
         x = self.avgpool(self.M_layer3(x))
         #print(x.size())         
         out = x.view(-1, 256)
         out_1 = self.fc_1(out)
         out_2 = self.fc_2(out)
         return out_1, out_2








