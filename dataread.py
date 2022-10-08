


import numpy as np

def normalization(data):   ########def定义归一化函数，主要的目的：使模型更快收敛
    #_range = np.max(data) - np.min(data)
    #return (data - np.min(data)) / _range

    _range = np.max(abs(data))      ######找到图像中绝对值的最大值
    return data / _range    #######数据除以最大值，使数据在[-1,1]之间，让模型收敛更快，使训练效果更好

def batch_data_read(data_path):
    data_all = []
    labels_1 = []
    labels_2 = []
    for i in range(len(data_path)):
        data = np.loadtxt(data_path[i][0])
        data = data[:708]
        #print(data.shape)
        if len(data) < 700:
           continue
        label_1 = int(data_path[i][1])
        label_2 = int(data_path[i][2])
        data_all.append(data)
        #label = data[i][1]
        labels_1.append(label_1)   
        labels_2.append(label_2)
        #print(data.shape,label)
    return np.array(data_all), np.array(labels_1), np.array(labels_2)

"""
data = np.loadtxt("./train.txt", dtype=str, delimiter=',')
a,b,c = batch_data_read(data)
print(a.shape,b.shape)
"""








