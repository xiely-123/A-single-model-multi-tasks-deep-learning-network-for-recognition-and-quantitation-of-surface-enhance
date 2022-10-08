# -*- coding: utf-8 -*-
"""
Created on Tue Dec  1 10:11:41 2020

@author: ALW
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import model as ml
from torch import optim
import dataread
from torch.autograd import Variable
import time
import os
#import loss
import random

batch_size = 128
batch_size1 = 500
lr = 1e-3
os.environ['CUDA_VISIBLE_DEVICES'] = '0'   ####gpu选择

net = ml.ResNet(1,3,2)
#net = torch.load("/data/nfs_rt16/luyuan/code/interspeech_classifition/ResNet_0207_augment/7_my_model.pth")#.module
#print(net)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  ###如果没有gpu选择cpu

if torch.cuda.device_count() > 1:
  net = nn.DataParallel(net)  ####gpu并行训练
# Assuming that we are on a CUDA machine, this should print a CUDA device:
net.to(device)  ####网络采用gpu
print(device)
net=net.double()  
criterion_re = torch.nn.CrossEntropyLoss()#torch.nn.MSELoss()  #####CE准则loss
criterion = torch.nn.CrossEntropyLoss()  #####CE准则loss
#criterion = torch.nn.SmoothL1Loss()
#criterion = loss.FocalLoss(5)
optimizer = optim.Adam(net.parameters(), lr=lr)  ########优化器
Train_list = np.loadtxt("./train_real.txt", dtype=str, delimiter=',')
devel_list = np.loadtxt("./test_real.txt", dtype=str, delimiter=',')
#Train_list = Train_list[1:]
#devel_list = devel_list[1:]
#devel_list1 = devel_list[1:500]
#devel_list = devel_list[500:]
#Train_list = np.concatenate([Train_list,devel_list],0)
print(len(Train_list),len(devel_list))


all_time = 0          #####一共所用时间
num_epochs = 100   #######训练轮数

#print(net)
#net.load_state_dict(checkpoint)

start_epoch = 0
for epoch in range(start_epoch,num_epochs):
    start_time = time.time()
    permutation = np.random.permutation(Train_list.shape[0])####数据随机
    shuffled_dataset = Train_list[permutation, :]
    if epoch%2 == 0:   ####学习率衰减
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
            lr = lr * 0.9
    net.train()
    for i in range(int(len(Train_list)/batch_size)+1):
        running_corrects = torch.zeros(1).squeeze().cuda()
        data_path = shuffled_dataset[i*batch_size:i*batch_size+batch_size]
        data,label,label_1 = dataread.batch_data_read(data_path)
        #print(data.shape,label_1)
        data = data.reshape((-1,1,708))
        inputs1 = Variable(torch.from_numpy(data)).to(device)  #######Train
        target = Variable(torch.from_numpy(label)).to(device)
        target_1 = Variable(torch.from_numpy(label_1)).to(device)        
        target = target.long()#.double
        target_1 = target_1.long()
        out,out1 = net(inputs1)
        #print(out.shape,target.shape)
        #out = out.reshape((-1))
        loss = criterion_re(out,target) #####out*inputs1.double()
        loss1 = criterion(out1,target_1)
        loss = 0.5*loss+0.5*loss1
        _ , prediction =  torch.max(out,1)
        _ , prediction1 =  torch.max(out1,1)
        #print(prediction1)
        running_corrects = torch.sum(prediction == target)
        running_corrects1 = torch.sum(prediction1 == target_1)
        optimizer.zero_grad()           #归零梯度，每次反向传播前都要归零梯度，不然梯度会累积，造成结果不收敛
        loss.backward()                 #反向传播
        optimizer.step()                #更新参数
        train_time = time.time() - start_time 
        print('\r','Epoch[{}/{}],Process[{}/{}],loss:{:.6f},ACC5:{:.6f},ACC2:{:.6f},time:{:.3f},leanring_rate:{:.6f}'.format(epoch + 1, num_epochs, i + 1, int(len(Train_list)/batch_size)+1, loss.item(), running_corrects/len(label),running_corrects1/len(label_1), train_time, param_group['lr']),end='')
    running_corrects = torch.zeros(1).squeeze().cuda()
    running_corrects1 = torch.zeros(1).squeeze().cuda()
    
    net.eval()
    for i in range(1):
        data_path = devel_list[i*batch_size1:i*batch_size1+batch_size1]
        data,label,label_1 = dataread.batch_data_read(data_path) 
        data = data.reshape((-1,1,708))
        inputs1 = Variable(torch.from_numpy(data)).to(device)  #######Train
        target = Variable(torch.from_numpy(label)).to(device)  #######输入phase*cos
        target_1 = Variable(torch.from_numpy(label_1)).to(device)        
        target = target.long()
        target_1 = target_1.long()
        out,out1 = net(inputs1)
        _ , prediction =  torch.max(out,1)
        #prediction =  torch.round(out.reshape((-1)))
        _ , prediction1 =  torch.max(out1,1)
        #print(prediction.shape,target.shape)
        running_corrects = torch.sum(prediction == target)
        running_corrects1 = torch.sum(prediction1 == target_1)
        #print(running_corrects,running_corrects1)
    print("         ")
    print('Accuracy5: %f'%((running_corrects/len(devel_list)).cpu().detach().data.numpy()))
    print('Accuracy2: %f'%((running_corrects1/len(devel_list)).cpu().detach().data.numpy()))
    #print('UAR: %f'%(UAR.cpu().detach().data.numpy()))
    
    
    #torch.save(net, "/data/nfs_rt16/luyuan/code/interspeech_classifition/SKnet_0224_noresize_0.4/"+str(epoch + 1)+"_my_model.pth")  # 保存模型
