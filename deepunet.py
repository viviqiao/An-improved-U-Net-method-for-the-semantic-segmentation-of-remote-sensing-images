# coding=utf-8
# import matplotlib
from keras import metrics
# matplotlib.use("Agg")
import matplotlib.pyplot as plt
import argparse
import numpy as np
import keras
import math
from keras.optimizers import *
from keras.layers import Conv2DTranspose
from keras.layers.core import Dense, Dropout, Activation
from keras.models import Sequential
from keras.regularizers import l2
from keras.layers.pooling import AveragePooling2D, GlobalAveragePooling2D
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D, BatchNormalization, Reshape, Permute, Activation, Input, \
    merge
from keras.utils.np_utils import to_categorical
from keras.layers.convolutional import Convolution2D, ZeroPadding2D
from keras.preprocessing.image import img_to_array
from keras.callbacks import ModelCheckpoint
from sklearn.preprocessing import LabelEncoder
from keras.models import Model
from keras.layers.merge import concatenate
from PIL import Image
import keras.backend as K
import matplotlib.pyplot as plt

import cv2
import random
import os
import tensorflow as tf
from tqdm import tqdm
import keras.backend as K
import pickle
from keras.callbacks import LearningRateScheduler
from libtiff import TIFF
from keras.utils import plot_model

os.environ['CUDA_VISIBLE_DEVICES'] = "1"
'''
gpus = tf.config.experimental.list_physical_devices("GPU")
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
else:
    print('erro')

config = tf.compat.v1.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.88  # CUDA_ERROR_OUT_OF_MEMORY
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
'''
seed = 7
np.random.seed(seed)
global H
# data_shape = 360*480
img_w = 256
img_h = 256
# 有一个为背景
n_label = 6
# n_label = 1

classes = [0., 1., 2., 3., 4., 5.]

labelencoder = LabelEncoder()
labelencoder.fit(classes)

image_sets = ['1.png', '2.png', '3.png']

trainsrc = '../data/'
trainlab = '../mask/'


def code(label):
    c1 = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    c2 = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0]
    c3 = [0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
    c4 = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0]
    c5 = [0.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    c6 = [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
    train_lable = np.zeros([256, 256, 6])
    for i in range(256):
        for j in range(256):
            if label[i][j] == 29:
                train_lable[i][j] = c1
            elif label[i][j] == 76:
                train_lable[i][j] = c2
            elif label[i][j] == 150:
                train_lable[i][j] = c3
            elif label[i][j] == 179:
                train_lable[i][j] = c4
            elif label[i][j] == 226:
                train_lable[i][j] = c5
            elif label[i][j] == 255:
                train_lable[i][j] = c6

            else:
                list = []
                list.append(abs(label[i][j] - 29))
                list.append(abs(label[i][j] - 76))
                list.append(abs(label[i][j] - 150))
                list.append(abs(label[i][j] - 179))
                list.append(abs(label[i][j] - 226))
                list.append(abs(label[i][j] - 255))
                index = list.index(min(list))
                if index == 0:
                    train_lable[i][j] = c1
                elif index == 1:
                    train_lable[i][j] = c2
                elif index == 2:
                    train_lable[i][j] = c3
                elif index == 3:
                    train_lable[i][j] = c4
                elif index == 4:
                    train_lable[i][j] = c5
                elif index == 5:
                    train_lable[i][j] = c6

    return train_lable


def load_img(path, grayscale=False):
    if grayscale:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img = np.array(img, dtype="float")
    else:
        img = cv2.imread(path)
        img = np.array(img, dtype="float") / 255.0
    return img


# filepath ='./unet_train/'

def get_train_val(val_rate=0.25):
    train_url = []
    train_set = []
    val_set = []
    for pic in os.listdir(trainsrc):
        train_url.append(pic)
    random.shuffle(train_url)
    total_num = len(train_url)
    val_num = int(val_rate * total_num)
    for i in range(len(train_url)):
        if i < val_num:
            val_set.append(train_url[i])
        else:
            train_set.append(train_url[i])
    return train_set, val_set


# data for training
def generateData(batch_size, data=[]):
    # print 'generateData...'
    while True:
        train_data = []
        train_label = []
        batch = 0
        for i in (range(len(data))):
            url = data[i]
            batch += 1
            img = load_img(trainsrc + url)
            img = img_to_array(img)
            train_data.append(img)
            label = load_img(trainlab + url, grayscale=True)
            label = img_to_array(label)
            # label=label/255.0
            label = code(label)
            # label = cv2.cvtColor(label, cv2.COLOR_GRAY2BGR)
            train_label.append(label)
            if batch % batch_size == 0:
                # print 'get enough bacth!\n'
                train_data = np.array(train_data)
                train_label = np.array(train_label)
                yield (train_data, train_label)
                train_data = []
                train_label = []
                batch = 0

            # data for validation


def generateValidData(batch_size, data=[]):
    # print 'generateValidData...'
    while True:
        valid_data = []
        valid_label = []
        batch = 0
        for i in (range(len(data))):
            url = data[i]
            batch += 1
            img = load_img(trainsrc + url)
            img = img_to_array(img)
            valid_data.append(img)
            label = load_img(trainlab + url, grayscale=True)
            label = img_to_array(label)
            # label=label/255.0
            label = code(label)
            # label = cv2.cvtColor(label, cv2.COLOR_GRAY2BGR)
            valid_label.append(label)
            if batch % batch_size == 0:
                valid_data = np.array(valid_data)
                valid_label = np.array(valid_label)
                yield (valid_data, valid_label)
                valid_data = []
                valid_label = []
                batch = 0


def conv_block(ip, nb_filter, dropout_rate=None, weight_decay=1E-4):
    x = Activation('relu')(ip)
    x = Convolution2D(nb_filter, 3, 3, border_mode="same", bias=False,
                      W_regularizer=l2(weight_decay))(x)
    if dropout_rate:
        x = Dropout(dropout_rate)(x)
    return x


def _conv2d_same(x, filters, prefix, padd, stride=1, kernel_size=3, rate=1, active=None):
    if stride == 1:
        x = Conv2D(filters,
                   W_regularizer=l2(1E-4),
                   kernel_size=kernel_size,
                   strides=(stride, stride),
                   padding=padd, use_bias=False,
                   dilation_rate=(rate, rate),
                   kernel_initializer="TruncatedNormal",
                   activation=active,
                   name=prefix)(x)
        return x
    else:
        kernel_size_effective = kernel_size + (kernel_size - 1) * (rate - 1)
        pad_total = kernel_size_effective - 1
        pad_beg = pad_total // 2
        pad_end = pad_total - pad_beg
        x = ZeroPadding2D((pad_beg, pad_end))(x)
        x = Conv2D(filters,
                   (kernel_size, kernel_size),
                   strides=(stride, stride),
                   padding=padd, use_bias=False,
                   kernel_initializer="TruncatedNormal",
                   dilation_rate=(rate, rate),
                   activation=active,
                   name=prefix)(x)
        x = Dropout(0.4)(x)
    return x


def scheduler(epoch):
    if (epoch % 1 == 0 and epoch != 0):
        model.save('deepunet' + str(epoch) + '.h5')
    return K.get_value(model.optimizer.lr).astype('float32')


def unet(pretrained_weights=None, input_size=(256, 256, 3)):
    inputs = Input(shape=(img_w, img_h, 3), batch_shape=(None, img_w, img_h, 3))
    caxis = 1 if K.image_data_format() == 'channels_first' else -1
    x = BatchNormalization(mode=0, axis=caxis, gamma_regularizer=l2(1E-4), beta_regularizer=l2(1E-4))(inputs)
    convadd1 = Conv2D(8, 3, activation='relu', padding='same', kernel_initializer='he_normal')(x)
    convadd1 = Conv2D(8, 3, activation='relu', padding='same', kernel_initializer='he_normal')(convadd1)
    pooladd1 = MaxPooling2D(pool_size=(2, 2))(convadd1)


    convadd2 = Conv2D(16, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pooladd1)
    convadd2 = Conv2D(16, 3, activation='relu', padding='same', kernel_initializer='he_normal')(convadd2)
    pooladd2 = MaxPooling2D(pool_size=(2, 2))(convadd2)

    convadd3 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pooladd2)
    convadd3 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(convadd3)
    pooladd3 = MaxPooling2D(pool_size=(2, 2))(convadd3)


    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pooladd3)
    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
    conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
    conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5)

    up6 = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(drop5))
    merge6 = concatenate([drop4, up6], axis=caxis)
    conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
    conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)

    up7 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv6))
    merge7 = concatenate([conv3, up7], axis=caxis)
    conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
    conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)

    up8 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv7))
    merge8 = concatenate([conv2, up8], axis=caxis)
    conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
    conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)

    up9 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv8))
    merge9 = concatenate([conv1, up9], axis=caxis)
    conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
    conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)

    up10 = Conv2D(32, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv9))
    merge10 = concatenate([convadd3, up10], axis=caxis)
    conv10 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge10)
    conv10 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv10)

    up11 = Conv2D(16, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv10))
    merge11 = concatenate([convadd2, up11], axis=caxis)
    conv11 = Conv2D(16, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge11)
    conv11 = Conv2D(16, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv11)
    up12 = Conv2D(8, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv11))
    merge12 = concatenate([convadd1, up12], axis=caxis)
    conv12 = Conv2D(8, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge12)
    conv12 = Conv2D(6, 3, padding='same', activation='softmax')(conv12)

    model = Model(input=inputs, output=conv12)

    model.compile(optimizer=Adam(lr=1e-3), loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    K.set_value(model.optimizer.lr, 0.001)
    if (pretrained_weights):
        model.load_weights(pretrained_weights)

    return model


def unet1():
    inputs = Input(shape=(img_w, img_h, 3), batch_shape=(None, img_w, img_h, 3))
    caxis = 1 if K.image_data_format() == 'channels_first' else -1
    inputs = Input(shape=(img_w, img_h, 3), batch_shape=(None, img_w, img_h, 3))
    convadd3 = BatchNormalization(momentum=0.99)(inputs)
    convadd3 = Conv2D(8, (3, 3), activation="relu", padding="same", kernel_initializer='he_normal')(convadd3)
    convadd3 = BatchNormalization(momentum=0.99)(convadd3)
    convadd3 = Conv2D(8, (1, 1), activation="relu", padding="same", kernel_initializer='he_normal')(convadd3)
    convadd3 = Dropout(0.02)(convadd3)
    pooladd3 = AveragePooling2D(pool_size=(2, 2))(convadd3)

    convadd1 = BatchNormalization(momentum=0.99)(pooladd3)
    convadd1 = Conv2D(16, (3, 3), activation="relu", padding="same", kernel_initializer='he_normal')(convadd1)
    convadd1 = BatchNormalization(momentum=0.99)(convadd1)
    convadd1 = Conv2D(16, (1, 1), activation="relu", padding="same", kernel_initializer='he_normal')(convadd1)
    convadd1 = Dropout(0.02)(convadd1)
    pooladd1 = AveragePooling2D(pool_size=(2, 2))(convadd1)

    convadd2 = BatchNormalization(momentum=0.99)(pooladd1)
    convadd2 = Conv2D(32, (3, 3), activation="relu", padding="same", kernel_initializer='he_normal')(convadd2)
    convadd2 = BatchNormalization(momentum=0.99)(convadd2)
    convadd2 = Conv2D(32, (1, 1), activation="relu", padding="same", kernel_initializer='he_normal')(convadd2)
    convadd2 = Dropout(0.02)(convadd2)
    pooladd2 = AveragePooling2D(pool_size=(2, 2))(convadd2)

    conv1 = BatchNormalization(momentum=0.99)(pooladd2)
    conv1 = Conv2D(64, (3, 3), activation="relu", padding="same", kernel_initializer='he_normal')(conv1)
    conv1 = BatchNormalization(momentum=0.99)(conv1)
    conv1 = Conv2D(64, (1, 1), activation="relu", padding="same", kernel_initializer='he_normal')(conv1)
    conv1 = Dropout(0.02)(conv1)
    pool1 = AveragePooling2D(pool_size=(2, 2))(conv1)

    pool1 = BatchNormalization(momentum=0.99)(pool1)
    conv2 = Conv2D(128, (3, 3), activation="relu", padding="same", kernel_initializer='he_normal')(pool1)
    conv2 = BatchNormalization(momentum=0.99)(conv2)
    conv2 = Conv2D(128, (1, 1), activation="relu", padding="same", kernel_initializer='he_normal')(conv2)
    conv2 = Dropout(0.02)(conv2)
    pool2 = AveragePooling2D(pool_size=(2, 2))(conv2)

    pool2 = BatchNormalization(momentum=0.99)(pool2)
    conv3 = Conv2D(256, (3, 3), activation="relu", padding="same", kernel_initializer='he_normal')(pool2)
    conv3 = BatchNormalization(momentum=0.99)(conv3)
    conv3 = Conv2D(256, (1, 1), activation="relu", padding="same", kernel_initializer='he_normal')(conv3)
    conv3 = Dropout(0.02)(conv3)
    pool3 = AveragePooling2D(pool_size=(2, 2))(conv3)

    pool3 = BatchNormalization(momentum=0.99)(pool3)
    conv4 = Conv2D(512, (3, 3), activation="relu", padding="same", kernel_initializer='he_normal')(pool3)
    conv4 = BatchNormalization(momentum=0.99)(conv4)
    conv4 = Conv2D(512, (1, 1), activation="relu", padding="same", kernel_initializer='he_normal')(conv4)
    conv4 = Dropout(0.02)(conv4)
    pool4 = AveragePooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(1024, (3, 3), activation="relu", padding="same", kernel_initializer='he_normal')(pool4)
    conv5 = Conv2D(1024, (1, 1), activation="relu", padding="same", kernel_initializer='he_normal')(conv5)

    conv6 = Conv2D(512, (3, 3), activation="relu", padding="same", kernel_initializer='he_normal')(conv5)
    up6 = concatenate([UpSampling2D(size=(2, 2))(conv5), conv4], axis=3)
    conv6 = Conv2D(512, (3, 3), activation="relu", padding="same", kernel_initializer='he_normal')(up6)

    conv7 = Conv2D(256, (3, 3), activation="relu", padding="same", kernel_initializer='he_normal')(conv6)
    up7 = concatenate([UpSampling2D(size=(2, 2))(conv6), conv3], axis=3)
    conv7 = Conv2D(256, (3, 3), activation="relu", padding="same", kernel_initializer='he_normal')(up7)

    conv8 = Conv2D(128, (3, 3), activation="relu", padding="same", kernel_initializer='he_normal')(conv7)
    up8 = concatenate([UpSampling2D(size=(2, 2))(conv7), conv2], axis=3)
    conv8 = Conv2D(128, (3, 3), activation="relu", padding="same", kernel_initializer='he_normal')(up8)

    conv9 = Conv2D(64, (3, 3), activation="relu", padding="same", kernel_initializer='he_normal')(conv8)
    up9 = concatenate([UpSampling2D(size=(2, 2))(conv8), conv1], axis=3)
    conv9 = Conv2D(64, (3, 3), activation="relu", padding="same", kernel_initializer='he_normal')(up9)

    conv10 = Conv2D(32, (3, 3), activation="relu", padding="same", kernel_initializer='he_normal')(conv9)
    up10 = concatenate([UpSampling2D(size=(2, 2))(conv9), convadd2], axis=3)
    conv10 = Conv2D(32, (3, 3), activation="relu", padding="same", kernel_initializer='he_normal')(up10)

    conv11 = Conv2D(16, (3, 3), activation="relu", padding="same", kernel_initializer='he_normal')(conv10)
    up11 = concatenate([UpSampling2D(size=(2, 2))(conv10), convadd1], axis=3)
    conv11 = Conv2D(16, (3, 3), activation="softmax", padding="same", kernel_initializer='he_normal')(up11)

    conv12 = Conv2D(8, (3, 3), activation="relu", padding="same", kernel_initializer='he_normal')(conv11)
    up12 = concatenate([UpSampling2D(size=(2, 2))(conv11), convadd3], axis=3)
    conv12 = Conv2D(6, (3, 3), activation="softmax", padding="same", kernel_initializer='he_normal')(up12)

    model = Model(inputs=inputs, outputs=conv12)
    model.compile(optimizer=Adam(lr=1e-3), loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary();
    K.set_value(model.optimizer.lr, 0.01)
    return model


def unet2():
    inputs = Input(shape=(img_w, img_h, 3), batch_shape=(None, img_w, img_h, 3))
    conv1 = Conv2D(8, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    pool1 = AveragePooling2D(pool_size=(2, 2))(conv1)  # 16

    conv2 = BatchNormalization(momentum=0.99)(pool1)
    conv2 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
    conv2 = BatchNormalization(momentum=0.99)(conv2)
    conv2 = Conv2D(64, 1, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
    conv2 = Dropout(0.02)(conv2)
    pool2 = AveragePooling2D(pool_size=(2, 2))(conv2)  # 8

    conv3 = BatchNormalization(momentum=0.99)(pool2)
    conv3 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
    conv3 = BatchNormalization(momentum=0.99)(conv3)
    conv3 = Conv2D(128, 1, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
    conv3 = Dropout(0.02)(conv3)
    pool3 = AveragePooling2D(pool_size=(2, 2))(conv3)  # 4

    conv4 = BatchNormalization(momentum=0.99)(pool3)
    conv4 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
    conv4 = BatchNormalization(momentum=0.99)(conv4)
    conv4 = Conv2D(256, 1, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
    conv4 = Dropout(0.02)(conv4)
    pool4 = AveragePooling2D(pool_size=(2, 2))(conv4)

    conv5 = BatchNormalization(momentum=0.99)(pool4)
    conv5 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
    conv5 = BatchNormalization(momentum=0.99)(conv5)
    conv5 = Conv2D(512, 1, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
    conv5 = Dropout(0.02)(conv5)
    pool4 = AveragePooling2D(pool_size=(2, 2))(conv4)
    # conv5 = Conv2D(35, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
    # drop4 = Dropout(0.02)(conv5)
    pool4 = AveragePooling2D(pool_size=(2, 2))(pool3)  # 2
    pool5 = AveragePooling2D(pool_size=(2, 2))(pool4)  # 1

    conv6 = BatchNormalization(momentum=0.99)(pool5)
    conv6 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)

    conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)
    up7 = (UpSampling2D(size=(2, 2))(conv7))  # 2
    conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(up7)
    merge7 = concatenate([pool4, conv7], axis=3)

    conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
    up8 = (UpSampling2D(size=(2, 2))(conv8))  # 4
    conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(up8)
    merge8 = concatenate([pool3, conv8], axis=3)

    conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
    up9 = (UpSampling2D(size=(2, 2))(conv9))  # 8
    conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(up9)
    merge9 = concatenate([pool2, conv9], axis=3)

    conv10 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
    up10 = (UpSampling2D(size=(2, 2))(conv10))  # 16
    conv10 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(up10)

    conv11 = Conv2D(16, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv10)
    up11 = (UpSampling2D(size=(2, 2))(conv11))  # 32
    conv11 = Conv2D(16, 3, activation='relu', padding='same', kernel_initializer='he_normal')(up11)

    conv12 = Conv2D(6, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv11)
    conv12 = Conv2D(6, 3, activation='softmax', padding='same', kernel_initializer='he_normal')(conv12)

    model = Model(input=inputs, output=conv12)
    print(model.summary())
    model.compile(optimizer=Adam(lr=1e-3), loss='categorical_crossentropy', metrics=['accuracy'])
    K.set_value(model.optimizer.lr, 0.001)
    return model


def train(args):
    EPOCHS = 80
    BS = 16
    global model
    # model = AtrousDenseUDeconvNet()
    # model1=keras.models.load_model('AtrousDense-U-DeconvNet155.h5')
    model = unet()
    keras.utils.plot_model(model, "deepunetModel.png", show_shapes=True)
    # model.load_weights('AtrousDense-U-DeconvNet155.h5',by_name=True)
    # model1.sumarray()
    # plot_model(model, to_file='model.png')
    train_set, val_set = get_train_val()
    train_numb = len(train_set)
    valid_numb = len(val_set)
    print("the number of train data is", train_numb)
    print("the number of val data is", valid_numb)
    callbacks_list = [

        keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=2),
        keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=2, verbose=0, mode='auto',
                                          min_lr=0.00000001),
        LearningRateScheduler(scheduler, verbose=1),
        keras.callbacks.TensorBoard(log_dir='deepunetlog1', update_freq=1000)
    ]
    global H
    H = model.fit_generator(generator=generateData(BS, train_set),
                            steps_per_epoch=train_numb // BS,
                            epochs=EPOCHS,
                            verbose=1,
                            validation_data=generateValidData(BS, val_set),
                            validation_steps=valid_numb // BS,
                            callbacks=callbacks_list,
                            max_q_size=10,
                            shuffle=True,
                            initial_epoch=0
                            )

    # plot the training loss and accuracy

    with open('hist.pickle', 'wb') as file_pi:
        pickle.dump(H.history, file_pi)
    #plt.style.use("ggplot")
    #plt.figure()
    try:
        model.save('deepunet.h5')
        #plt.style.use("ggplot")
        #plt.figure()
        '''
        N = EPOCHS        
        plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
        plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
        plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
        plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
        plt.title("Training Loss and Accuracy on AtrousDenseUDeconvNet Satellite Seg")
        plt.xlabel("Epoch #")
        plt.ylabel("Loss/Accuracy")
        plt.legend(loc="lower left")
        # plt.savefig(args["plot"])
        # plt.savefig('plot.png')
        plt.savefig('unet.png')
        '''
        with open('unet.txt', 'wb') as file_pi:
            pickle.dump(H.history, file_pi)
    except:
        model.save('unet.h5')
        with open('unettrainHistoryDict1.txt', 'wb') as file_pi:
            pickle.dump(H.history, file_pi)
    else:
        model.save('unet.h5')
        with open('unethis.txt', 'wb') as file_pi:
            pickle.dump(H.history, file_pi)


def args_parse():
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--data", help="training data's path",
                    required=False)
    ap.add_argument("-m", "--model", required=False,
                    help="path to output model")
    ap.add_argument("-p", "--plot", type=str, default="plot.png",
                    help="path to output accuracy/loss plot")
    args = vars(ap.parse_args())
    return args


if __name__ == '__main__':
    args = args_parse()
    filepath = args['data']
    train(args)
    # predict()
