import pandas as pd
import numpy as np
import os

dir = r"E:\等离子体泡新数据\csv文件\val"
dir1 = "E:\\等离子体泡新数据\\csv文件\\val"
listdir = os.listdir(dir)
print(len(listdir))
train_x = np.zeros(1)
train1 = np.zeros(1)
train_y = np.zeros(1)
for i in listdir:
    print('正在读取' + i)
    path    = os.path.join(dir1 , i)
    csv     = pd.read_csv(path)
    csv     = np.array(csv)
    # csv     = np.diff(csv,0)
    #print(csv.shape)
    x_train = csv[:, 0]
    #train = csv[:, 0]
    #train = train.reshape((len(train), 1))
    print(len(x_train))
    x_train = x_train.reshape((len(x_train), 1))
    # print(x_train)
    # print('diff前',x_train.shape)
    x_train = np.diff(x_train, axis=0)
    # print(x_train)
    # print('diff后',x_train.shape)
    x_train = np.vstack((np.zeros(1), x_train))
    #print(x_train.shape,y_tr)
    y_train = csv[:,1]
    y_train = y_train.reshape((len(y_train), 1))
    #print(x_train.shape, y_train.shape)
    # y_train = np.diff(y_train, 0)
    train_x = np.vstack((train_x,x_train))
    train_y = np.vstack((train_y, y_train))
    #train1 = np.vstack((train1,train))

train_x=np.delete(train_x,0,0)
#train = np.delete(train1,0,0)
print(train_x.shape)
train_y=np.delete(train_y,0,0)
print(train_x.shape,train_y.shape)
# np.save(r"E:\等离子泡\dataset\test2004night\2004test_x_night.npy",train_x)
# np.save(r"E:\等离子泡\dataset\test2004night\2004test_y_night_label.npy",train_y)
np.save(r"E:\等离子体泡新数据\csv文件\val_x.npy",train_x)
np.save(r"E:\等离子体泡新数据\csv文件\val_y.npy",train_y)











































