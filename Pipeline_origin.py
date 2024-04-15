import os
import numpy as np
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
import tensorflow.keras as K
from Data_origin import generator_conv1d
from Unet_model import Unet_model
from IncUnet import IncUnet
import tensorflow as tf
#from sklearn.metrics import classification_report, f1_score, recall_score, precision_score
from scipy.ndimage import distance_transform_edt as distance
import datetime
import time
from eval import detect, correction

gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
print(gpus)
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# tf.config.experimental.set_virtual_device_configuration(
#     gpus[0],
#     [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=8000)]
# )

def re(y_true, y_pred):
    true_positives = K.backend.sum(K.backend.round(K.backend.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.backend.sum(K.backend.round(K.backend.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.backend.epsilon())
    # print(true_positives)
    # print(possible_positives)
    return recall


def prec(y_true, y_pred):
    true_positives = K.backend.sum(K.backend.round(K.backend.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.backend.sum(K.backend.round(K.backend.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.backend.epsilon())
    return precision


def f1(y_true, y_pred):
    precision = prec(y_true, y_pred)
    recall = re(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.backend.epsilon()))


class Pipeline(object):
    def __init__(self):
        self.windows = 1280
        self.batch_size = 16
        # self.ratio = 0.5
        self.model = 'ResIncUnet'
        self.lr = 0.001
        self.epoch = 50
        # self.thres = 0.5
        #self.evtList = _Data().read_csv(r'E:\Unet\time-series-segmentation-main\Automatic ICME detection\data\data\listOfICMEs.csv', index_col=None)

    def fit(self):
        train_data = generator_conv1d(mode='train', batch_size=self.batch_size,
                                      windows=self.windows, ratio=ratio)
        val_data = generator_conv1d(mode='val', batch_size=self.batch_size,
                                    windows=self.windows)

        callback = []
        # callbacks.append(ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5,
        #                                    epsilon=0.001, cooldown=1, verbose=1))
        callback.append(ModelCheckpoint(filepath="E:/等离子体泡新数据/csv文件/" +"/"+'model_{epoch:04d}_{val_f1:.04f}_{val_re:.04f}_{val_prec:.04f}_{val_loss:.04f}_{val_accuracy:.04f}_1.hdf5',
                                         monitor='val_f1', verbose=1,save_best_only=True,mode='max'))

        #callbacks.append(tf.keras.callbacks.TensorBoard(log_dir="fit_logss/", histogram_freq=1))

        model = Unet_model().ResIncUnet()
        #model = IncUnet().Incunet()
        model.compile(loss='binary_crossentropy', optimizer=optimizers.Adam(lr=self.lr),
                      metrics=[f1, re, prec, 'accuracy'])
        model.summary()

        # model = tf.keras.models.load_model(r"E:\Unet\ICME\result_origin_2\RUnet_2\0.7_64\model_0066_0.486455_0.878790_0.356936_0.3969_0.7685_1.hdf5",
        #                    custom_objects={'f1': f1, 're': re, 'prec': prec})
        print("开始训练")
        history=model.fit_generator(generator=train_data.__iter__(), epochs=self.epoch, steps_per_epoch=300,
                            verbose=1, validation_data=val_data.__iter__(), validation_steps=100,
                            callbacks=callback, initial_epoch=0)

        return model

    def test(self,):

        def f1(y_true, y_pred):
            precision = prec(y_true, y_pred)
            recall = re(y_true, y_pred)
            return 2 * ((precision * recall) / (precision + recall + K.backend.epsilon()))
        
        model = tf.keras.models.load_model(r"E:\等离子泡\原始公式对比_数据集\990102年数据训�?04年验证\Unet\0.7_1280\model_0001_0.9604_0.9330_0.9895_0.4527_0.9806_1.hdf5",custom_objects={'f1': f1, 're': re, 'prec': prec})

        
        print('--'*20)

        X_test = np.load(r"E:\等离子泡\原始公式对比_数据集\dayandnight\2003night\2003_night_x_diff.npy")
        Y_test = np.load(r"E:\等离子泡\原始公式对比_数据集\formula2004\formula2004_night_dataset\2004night_y_label.npy")
        test_para = X_test.copy()
        #test_label = Y_test.copy()
        print('*'*20)

        prediction = []
        for i in range(int(test_para.shape[0]/self.windows)):
            x_test = test_para[i * self.windows:(i + 1) * self.windows, :]
            x_test = x_test.reshape((1, self.windows, 1))
            a = model.predict(x_test)[0]
            #print(a)
            prediction.append(a)
            # print(i)
        # print(0, ':')
        x_test = test_para[-self.windows:, :]
        x_test = x_test.reshape((1, self.windows, 1))
        a = model.predict(x_test, verbose=1)[0]
        prediction.append(a[-(test_para.shape[0] - int(test_para.shape[0] / self.windows) * self.windows):, :])
        print('Finish!')

        prediction = np.concatenate(prediction, axis=0)
        for i in range(prediction.shape[0]):
            if prediction[i, 0] > 0.5:
                prediction[i, 0] = 1
            else:
                prediction[i, 0] = 0
        prediction = prediction.astype(int)

        pre_label = prediction.copy()
        pre_label = pre_label.reshape((pre_label.shape[0], 1)).astype(int)
        print('pre_label.shape',pre_label.shape)

        np.save(r"E:\等离子泡\交叉计算\test\unet_2003new_y_18.npy", pre_label)
        detect(Y_test,pre_label,0.5,True)
        


if __name__ == '__main__':
    ratio = 0.7
    pipeline = Pipeline()
    pipeline.fit()
