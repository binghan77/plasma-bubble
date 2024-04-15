import scipy.constants as constants
import pandas as pds
import datetime
import numpy as np
import pyarrow.parquet as pq
#from sklearn.preprocessing import StandardScaler
import os
import random
import tensorflow as tf
import keras as K

class _Data(object):
    def __init__(self):
        self.windows = 1280
        #self.evtList = self.read_csv(r'E:\Unet\time-series-segmentation-main\Automatic ICME detection\data\data\listOfICMEs.csv', index_col=None)

class generator_conv1d(object):
    def __init__(self, mode='train', batch_size=128, windows=1280, ratio=0.7):
        self.mode = mode
        self.batch_size = batch_size  # 第一个维度
        self.windows = windows  # 第二个维度
        self.ratio = ratio

        self.X_val = np.load(r"E:\\等离子体泡新数据\\csv文件\\val_x.npy")
        self.X_train = np.load(r"E:\\等离子体泡新数据\\csv文件\\train_x.npy")
        
        self.Y_val = np.load(r"E:\\等离子体泡新数据\\csv文件\\val_y.npy")
        self.Y_train = np.load(r"E:\\等离子体泡新数据\\csv文件\\train_y.npy")
        

        self.index_train_1, self.index_train_0 = self.get_index(self.Y_train)#训练集中1和0的分开
        self.index_val_1, self.index_val_0     = self.get_index(self.Y_val)#验证集中1和0的分开
        self.index_val                         = self.index_val_0 + self.index_val_1


    def __iter__(self):
        if self.mode == 'train':
            input_train, output_train = [], []
            while 1:
                x, y = self.get_train_data()
                input_train.append(x)
                output_train.append(y)

                if len(output_train) >= self.batch_size:
                    train_y = np.array(output_train)
                    train_x = np.array(input_train)
                    yield (train_x, train_y)
                    input_train, output_train = [], []
        else:
            input_val, output_val = [], []
            while 1:
                x, y = self.get_val_data()
                input_val.append(x)
                output_val.append(y)

                if len(output_val) >= self.batch_size:
                    val_y = np.array(output_val)
                    val_x = np.array(input_val)
                    yield (val_x, val_y)
                    input_val, output_val = [], []

    def get_train_data(self):
        sampling = random.random()
        #print('sampling',sampling)
        sampling_augment = random.random()
        #print('sampling_augment',sampling_augment)

        if sampling < self.ratio:
            index = random.sample(self.index_train_1, 1)[0]
            x = self.X_train[index - int(self.windows / 2):index + int(self.windows / 2), :]
            y = self.Y_train[index - int(self.windows / 2):index + int(self.windows / 2), :]

            #随机增强
            if sampling_augment < 1:
                x = x
                y = y
            else:
                # 数据扩充： 翻转
                x = np.flip(x, axis=0)
                y = np.flip(y, axis=0)
            
                # 数据扩充： 加噪声
                muti_noise = np.random.normal(1, 0.001, (x.shape[0], 1))
                x *= muti_noise
            
                add_noise = np.random.normal(0, 0.001, (x.shape[0], 1))
                x += add_noise
        else:
            index = random.sample(self.index_train_0, 1)[0]
            index = random.sample(self.index_train_1, 1)[0]
            x = self.X_train[index - int(self.windows / 2):index + int(self.windows / 2), :]
            y = self.Y_train[index - int(self.windows / 2):index + int(self.windows / 2), :]

        return x, y

    def get_val_data(self):
        index = random.sample(self.index_val, 1)[0]
        while (self.X_train[index - int(self.windows / 2), :] != [0] or self.X_train[index + int(self.windows / 2),:] != [0]):
            index = random.sample(self.index_train_1, 1)[0]
        x = self.X_train[index - int(self.windows / 2):index + int(self.windows / 2), :]
        y = self.Y_train[index - int(self.windows / 2):index + int(self.windows / 2), :]
        return x, y

    def get_index(self, file):
        index_1, index_0 = [], []

        for i in range(int(self.windows/2), file.shape[0]-int(self.windows/2)):
            if file[i, -1] == 1:
                index_1.append(i)
            else:
                index_0.append(i)

        return index_1, index_0

if __name__ == '__main__':
    data = _Data()
