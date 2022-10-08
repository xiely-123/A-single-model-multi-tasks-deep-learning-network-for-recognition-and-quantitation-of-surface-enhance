import numpy as np
import glob

data = glob.glob("./dataset/*/*.txt")
print(len(data))
a = np.loadtxt(data[0])
print(a.shape)

data_label = open('real_data.txt',mode='w')
for i in range(len(data)):
    if "-4" in data[i]:
       label = 0
    if "-5" in data[i]:
       label = 1
    if "-6" in data[i]:
       label = 2
    if "-7" in data[i]:
       label = 3
    if "-8" in data[i]:
       label = 4
    if "ROS-" in data[i]:
       label_1 = 0
    if "PHE-" in data[i]:
       label_1 = 1
    if "1v1" in data[i]:
       label_1 = 2
    if "1v4" in data[i]:
       label_1 = 3
    if "4v1" in data[i]:
       label_1 = 4

       

    data_label.write(data[i]+","+str(label)+","+str(label_1)+' \n')
data_label.close()


data = np.loadtxt("./real_data.txt", dtype=str, delimiter=',')
train=open('train_real_3.txt',mode='w')
test=open('test_real_3.txt',mode='w')
for i in range(len(data)):
    #print(i)
    if int(data[i][1]) == 1 or int(data[i][1]) == 3:
       #print(data[i],"test")
       test.write(data[i][0]+","+data[i][1]+","+data[i][2]+' \n')
    else:
       #print(data[i],"train")
       train.write(data[i][0]+","+data[i][1]+","+data[i][2]+' \n')
"""
import random
Test_0 = random.sample(range(0, 99), 20)
Test_1 = random.sample(range(100, 199), 20)
Test_2 = random.sample(range(200, 299), 20)
Test_3 = random.sample(range(300, 399), 20)
Test_4 = random.sample(range(400, 499), 20)
Test_5 = random.sample(range(500, 599), 20)

Test_6 = random.sample(range(600, 699), 20)
Test_7 = random.sample(range(700, 799), 20)
Test_8 = random.sample(range(800, 899), 20)
Test_9 = random.sample(range(900, 999), 20)
Test_10 = random.sample(range(1000, 1099), 20)
Test_11 = random.sample(range(1100, 1199), 20)
Test_12 = random.sample(range(1200, 1299), 20)
Test_13 = random.sample(range(1300, 1399), 20)
Test_14 = random.sample(range(1400, 1499), 20)
Test_15 = random.sample(range(1500, 1599), 20)
Test_16 = random.sample(range(1600, 1699), 20)
Test_17 = random.sample(range(1700, 1799), 20)
Test_18 = random.sample(range(1800, 1899), 20)
Test_19 = random.sample(range(1900, 1999), 20)
Test_20 = random.sample(range(2000, 2099), 20)
Test_21 = random.sample(range(2100, 2199), 20)
Test_22 = random.sample(range(2200, 2299), 20)
Test_23 = random.sample(range(2300, 2399), 20)
Test_24 = random.sample(range(2400, 2499), 20)


#Test_list = Test_0+Test_1+Test_2+Test_3+Test_4+Test_5+Test_6+Test_7+Test_8+Test_9+Test_10+Test_11+Test_12+Test_13+Test_14+Test_15+Test_16+Test_17+Test_18+Test_19+Test_20+Test_21+Test_22+Test_23+Test_24
Test_list = Test_0+Test_1+Test_2+Test_3+Test_4+Test_5
#print(Test_list)
for i in range(len(data)):
    if i in Test_list:
       #print(data[i],"test")
       test.write(data[i][0]+","+data[i][1]+","+data[i][2]+' \n')
    else:
       #print(data[i],"train")
       train.write(data[i][0]+","+data[i][1]+","+data[i][2]+' \n')

train.close()
test.close()
"""