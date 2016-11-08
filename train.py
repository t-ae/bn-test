#!/usr/bin/env python

from keras.models import Sequential, load_model
from keras.layers.core import Dense, Activation, Flatten
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.datasets import cifar10
from keras.callbacks import EarlyStopping, TensorBoard
from keras.utils import np_utils
from keras.utils.layer_utils import layer_from_config
import sys
import os

initial_weights_path = os.path.join(os.path.dirname(__file__), "./initial_weights.h5")

if(len(sys.argv)!=2):
    exit(-1)

mode = int(sys.argv[1])

if mode==0:
    before = False
    after = False
elif mode==1:
    logdir = "/tmp/bn_test/no_bn"
    before = False
    after = False
elif mode==2:
    logdir = "/tmp/bn_test/before_relu"
    before = True
    after = False
elif mode==3:
    logdir = "/tmp/bn_test/after_relu"
    before = False
    after = True
elif mode==4:
    logdir = "/tmp/bn_test/both"
    before = True
    after = True


if mode!=0:
    initial_model = load_model(initial_weights_path)

model = Sequential()


conv1 = Convolution2D(32, 5, 5,border_mode='same', input_shape=[32, 32, 3], name="conv1")

model.add(conv1)

if before:
    model.add(BatchNormalization())
model.add(Activation('relu'))
if after:
    model.add(BatchNormalization())

model.add(MaxPooling2D(border_mode='same'))# 16x16

conv2 = Convolution2D(64, 5, 5,border_mode='same', name="conv2")
model.add(conv2)

if before:
    model.add(BatchNormalization())
model.add(Activation('relu'))
if after:
    model.add(BatchNormalization())

model.add(MaxPooling2D(border_mode='same')) # 8x8

model.add(Flatten())

dense1 = Dense(1024, name="dense1")
model.add(dense1)

if before:
    model.add(BatchNormalization())
model.add(Activation('relu'))
if after:
    model.add(BatchNormalization())

dense2 = Dense(10, name="dense2", activation='softmax')
model.add(dense2)


if mode==0:
    model.save(initial_weights_path)
    print("weights saved")
else:
    print("load weights")
    conv1.set_weights(initial_model.get_layer("conv1").get_weights())
    conv2.set_weights(initial_model.get_layer("conv2").get_weights())
    dense1.set_weights(initial_model.get_layer("dense1").get_weights())
    dense2.set_weights(initial_model.get_layer("dense2").get_weights())


    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    size = 32
    numChannels = 3
    X_train = X_train.reshape([-1, size, size, numChannels]) / 255
    X_test = X_test.reshape([-1, size, size, numChannels]) / 255
    y_train = np_utils.to_categorical(y_train)
    y_test = np_utils.to_categorical(y_test)

    model.fit(X_train, y_train, 
            nb_epoch=20,
            validation_split=0.2,
            callbacks=[
                EarlyStopping(patience=2),
                TensorBoard(log_dir=logdir, histogram_freq=1)
            ])