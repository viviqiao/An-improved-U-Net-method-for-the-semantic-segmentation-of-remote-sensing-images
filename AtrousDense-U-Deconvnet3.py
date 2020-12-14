# coding=utf-8
#import matplotlib
from keras import metrics
#matplotlib.use("Agg")
import matplotlib.pyplot as plt
import argparse
import numpy as np
import tensorflow.keras
import math
from keras import utils, callbacks
from keras.layers import Conv2DTranspose
from keras.layers.core import Dense, Dropout, Activation
from keras.models import Sequential
from keras.regularizers import l2
from keras.layers.pooling import AveragePooling2D, GlobalAveragePooling2D
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D, BatchNormalization, Reshape, Permute, Activation, Input, merge
from keras.utils.np_utils import to_categorical
from keras.layers.convolutional import Convolution2D, ZeroPadding2D
from keras.preprocessing.image import img_to_array
from keras.callbacks import ModelCheckpoint
from sklearn.preprocessing import LabelEncoder
from keras.models import Model
from keras.layers.merge import concatenate
from PIL import Image
import tensorflow.keras.backend as K
import matplotlib.pyplot as plt
import cv2
import random
import os
import tensorflow as tf
from tqdm import tqdm
import pickle
from keras.callbacks import LearningRateScheduler
from libtiff import TIFF
from keras.utils import plot_model
from keras.utils import multi_gpu_model
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
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

trainsrc = '../data1/data/'
trainlab = '../data1/mask/'


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
                list =[]
                list.append(abs(label[i][j]-29))
                list.append(abs(label[i][j]-76))
                list.append(abs(label[i][j]-150))
                list.append(abs(label[i][j]-179))
                list.append(abs(label[i][j]-226))
                list.append(abs(label[i][j]-255))
                index=list.index(min(list))
                if index==0:
                    train_lable[i][j] = c1
                elif index == 1:
                    train_lable[i][j] = c2
                elif index == 2:
                    train_lable[i][j] = c3
                elif index==3:
                    train_lable[i][j] = c4
                elif index==4:
                    train_lable[i][j] = c5
                elif index == 5:
                    train_lable[i][j] = c6

    return train_lable


def load_img(path, grayscale=False):
    if grayscale:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
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


def transition_block1(ip, nb_filter, dropout_rate=None, weight_decay=1E-4):
    caxis = 1 if K.image_data_format() == 'channels_first' else -1
    x = Convolution2D(nb_filter, 1, 1, border_mode="same", bias=False, kernel_initializer="TruncatedNormal",
                      W_regularizer=l2(weight_decay),activation='relu')(ip)
    if dropout_rate:
        x = Dropout(dropout_rate)(x)
    x = AveragePooling2D((2, 2), strides=(2, 2))(x)
    #x = _conv2d_same(x, nb_filter, 'entry_flow_conv7', kernel_size=3, stride=2, rate=1, padd='valid',active='relu')
    x = BatchNormalization(mode=0, axis=caxis, gamma_regularizer=l2(weight_decay),
                           beta_regularizer=l2(weight_decay))(x)
    return x


def transition_block2(ip, nb_filter, dropout_rate=None, weight_decay=1E-4):
    caxis = 1 if K.image_data_format() == 'channels_first' else -1
    x = Convolution2D(nb_filter, 1, 1, border_mode="same", bias=False, kernel_initializer="TruncatedNormal",
                      W_regularizer=l2(weight_decay),name='tanb2C1',activation='relu')(ip)
    if dropout_rate:
        x = Dropout(dropout_rate)(x)
    x = AveragePooling2D((2, 2), strides=(2, 2))(x)
    #x = _conv2d_same(x, nb_filter, 'entry_flow_conv8', kernel_size=3, stride=2, rate=1, padd='valid',active='relu')
    x = BatchNormalization(mode=0, axis=caxis, gamma_regularizer=l2(weight_decay),
                           beta_regularizer=l2(weight_decay))(x)
    return x


def Dense_Block1(x):
    caxis = 1 if K.image_data_format() == 'channels_first' else -1
    feature_list = [x]  # 256
    x1 = _conv2d_same(x, 8, 'entry_flow_conv1', kernel_size=3, stride=1, rate=2, padd='same',active='relu')  # 242
    x2 = _conv2d_same(x1, 8, 'entry_flow_conv2', kernel_size=3, stride=1, rate=5, padd='same',active='relu')  # 220
    x3 = _conv2d_same(x2, 16, 'entry_flow_conv3', kernel_size=3, stride=1, rate=8, padd='same',active='relu')  # 188
    feature_list.append(x1)
    feature_list.append(x2)
    feature_list.append(x3)
    c1 = concatenate(feature_list,caxis)
    # x4 = _conv2d_same(c1, 16, 'entry_flow_conv1', kernel_size=3, stride=2, rate=1, padd='same')  # 252
    return c1


def Dense_Block2(x):
    caxis = 1 if K.image_data_format() == 'channels_first' else -1
    feature_list = [x]
    x1 = _conv2d_same(x, 32, 'entry_flow_conv4', kernel_size=3, stride=1, rate=2, padd='same',active='relu')  # 242
    x2 = _conv2d_same(x1, 32, 'entry_flow_conv5', kernel_size=3, stride=1, rate=5, padd='same',active='relu')  # 220
    x3 = _conv2d_same(x2, 64, 'entry_flow_conv6', kernel_size=3, stride=1, rate=8, padd='same',active='relu')  # 188
    feature_list.append(x1)
    feature_list.append(x2)
    feature_list.append(x3)
    c1 = concatenate(feature_list,caxis)
    # x4 = _conv2d_same(c1, 64, 'entry_flow_conv1', kernel_size=3, stride=2, rate=1, padd='same')  # 252
    return c1

'''
class CustomModel(keras.Model):
    def train_step(self, data):
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        x, y = data

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)  # Forward pass
            # Compute the loss value
            # (the loss function is configured in `compile()`)
            loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(y, y_pred)
        # Return a dict mapping metric names to current value
        with open('perstep.txt', 'a') as file_object:
            file_object.write({m.name: m.result() for m in self.metrics})

        return {m.name: m.result() for m in self.metrics}
'''

def AtrousDenseUDeconvNet():
    caxis = 1 if K.image_data_format() == 'channels_first' else -1
    input = Input(shape=(img_w, img_h, 3), batch_shape=(None, img_w, img_h, 3))  # 256
    # inputs=keras.layers.convolutional.ZeroPadding2D(padding=(0, 0), dim_ordering='default')(input)
    inputs1 = Conv2D(filters=8, kernel_size=3, name="initial_conv2D", bias=False, strides=1, activation="relu",
                     padding="same", kernel_initializer="TruncatedNormal", W_regularizer=l2(1E-4))(input)  # 256
    x = BatchNormalization(mode=0, axis=caxis, gamma_regularizer=l2(1E-4), beta_regularizer=l2(1E-4))(inputs1)  # 256
    dense1 = Dense_Block1(x)  # 188
    tran1 = transition_block1(dense1, 16, dropout_rate=0.25, weight_decay=1E-4)
    dense2 = Dense_Block2(tran1)  # 40
    #a1 = keras.layers.PReLU(alpha_initializer='TruncatedNormal', alpha_regularizer=None, alpha_constraint=None,
     #                       shared_axes=None)(dense2)
    tran2=transition_block2(dense2, 144, dropout_rate=0.25, weight_decay=1E-4)
    #pool = MaxPooling2D(pool_size=(2, 2))(a1)
    # 64 64*64
    conv1add=Conv2D(128, (3, 3), padding="same", kernel_initializer='he_normal', W_regularizer=l2(1E-4),name='conv1add',
                   activation="relu")(tran2)
    conv1 = Conv2D(128, (3, 3), padding="same", kernel_initializer='he_normal', W_regularizer=l2(1E-4),
                   activation="relu")(conv1add)  # 64
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)  # 32
    conv2add= Conv2D(256, (3, 3), padding="same", kernel_initializer='he_normal', W_regularizer=l2(1E-4),name='conv2add',
                   activation="relu")(pool1)
    conv2 = Conv2D(256, (3, 3), padding="same", kernel_initializer='he_normal', W_regularizer=l2(1E-4),
                   activation="relu")(conv2add)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)  # 16
    conv3add = Conv2D(512, (3, 3), padding="same", kernel_initializer='he_normal', W_regularizer=l2(1E-4),name='conv3add',
                   activation="relu")(pool2)
    conv3 = Conv2D(512, (3, 3), padding="same", kernel_initializer='he_normal', W_regularizer=l2(1E-4),
                   activation="relu")(conv3add)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)  # 8
    conv4 = Conv2D(1024, (3, 3), padding="same", kernel_initializer='he_normal', W_regularizer=l2(1E-4),
                   activation="relu")(pool3)  # 8
    up1 = concatenate([UpSampling2D(size=(2, 2))(conv4), conv3], axis=caxis)  # 16
    conv5add = Conv2D(512, (3, 3), padding="same", kernel_initializer='he_normal', W_regularizer=l2(1E-4),name='conv5add',
                   activation="relu")(up1)
    conv5 = Conv2D(512, (3, 3), padding="same", kernel_initializer='he_normal', W_regularizer=l2(1E-4),
                   activation="relu")(conv5add)  # 16
    up2 = concatenate([UpSampling2D(size=(2, 2))(conv5), conv2], axis=caxis)  # 32
    conv6add = Conv2D(256, (3, 3), padding="same", kernel_initializer='he_normal', W_regularizer=l2(1E-4),name='conv6add',
                   activation="relu")(up2)
    conv6 = Conv2D(256, (3, 3), padding="same", kernel_initializer='he_normal', W_regularizer=l2(1E-4),
                   activation="relu")(conv6add)  # 32
    up3 = concatenate([UpSampling2D(size=(2, 2))(conv6), conv1], axis=caxis)  # 64
    conv7add = Conv2D(128, (3, 3), padding="same", kernel_initializer='he_normal', W_regularizer=l2(1E-4),name='conv7add',
                   activation="relu")(up3)
    conv7 = Conv2D(128, (3, 3), padding="same", kernel_initializer='he_normal', W_regularizer=l2(1E-4),
                   activation="relu" )(conv7add)  # 64 全连接
    conv8 = Conv2DTranspose(filters=64, kernel_size=3, padding="same", strides=2, kernel_initializer="TruncatedNormal",
                            W_regularizer=l2(1E-4), activation="relu")(conv7)  # 128
    conv9 = Conv2DTranspose(filters=32, kernel_size=3, padding="same", strides=2, kernel_initializer="TruncatedNormal",
                            W_regularizer=l2(1E-4), activation="relu")(conv8)
    conv10 = Conv2DTranspose(filters=6, kernel_size=3, padding="same", strides=1, kernel_initializer="TruncatedNormal",
                             W_regularizer=l2(1E-4), activation="softmax")(conv9)
    model = Model(input=input, output=conv10)
    #model = multi_gpu_model(model, gpus=2)
    #loss = tf.optimizers.RMSprop
    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    K.set_value(model.optimizer.lr, 0.00001)
    utils.plot_model(model, "AtrousDense-U-DeconvNet_model.png", show_shapes=True)
    return model



def scheduler(epoch):
    if (epoch % 1 == 0):
        model.save('AtrousDense-U-DeconvNet2' + str(epoch) + '.h5')
    return K.get_value(model.optimizer.lr).astype('float32')


def train(args):
    EPOCHS = 80
    BS =15
    global model
    model = AtrousDenseUDeconvNet()
    #model1=keras.models.load_model('AtrousDense-U-DeconvNet155.h5')
    utils.plot_model(model, "model1.png", show_shapes=True)
    model.load_weights('AtrousDense-U-DeconvNet25.h5',by_name=True)
    #model1.sumarray()
    #plot_model(model, to_file='model.png')
    train_set, val_set = get_train_val()
    train_numb = len(train_set)
    valid_numb = len(val_set)
    print("the number of train data is", train_numb)
    print("the number of val data is", valid_numb)
    callbacks_list = [

        callbacks.EarlyStopping(monitor='val_accuracy', patience=4),
        callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=2, verbose=0, mode='auto',
                                          min_lr=0.0000000001),
        LearningRateScheduler(scheduler, verbose=1),
        callbacks.TensorBoard(log_dir='log2',update_freq=1000)
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
                            initial_epoch=6
                            )

    # plot the training loss and accuracy

    with open('hist.pickle', 'wb') as file_pi:
        pickle.dump(H.history, file_pi)
    plt.style.use("ggplot")
    plt.figure()
    try:
        model.save('AtrousDense-U-DeconvNet.h5')
        plt.style.use("ggplot")
        plt.figure()
        N = EPOCHS
        plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
        plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
        plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
        plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
        plt.title("Training Loss and Accuracy on AtrousDense-U-DeconvNet Satellite Seg")
        plt.xlabel("Epoch #")
        plt.ylabel("Loss/Accuracy")
        plt.legend(loc="lower left")
        # plt.savefig(args["plot"])
        plt.savefig('plot.png')
        with open('trainHistoryDict1.txt', 'wb') as file_pi:
            pickle.dump(H.history, file_pi)
    except:
        model.save('AtrousDense-U-DeconvNet.h5')
        with open('trainHistoryDict1.txt', 'wb') as file_pi:
            pickle.dump(H.history, file_pi)
    else:
        model.save('AtrousDense-U-DeconvNet.h5')
        with open('trainHistoryDict1.txt', 'wb') as file_pi:
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
    
