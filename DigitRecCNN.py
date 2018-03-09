#!/usr/bin/env python3.5
# coding=utf-8

'''
@date = '17/12/1'
@author = 'lynnchan'
@email = 'chenliang@moutum.com'
'''

import csv_reader
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Binarizer

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation,Flatten
from keras.optimizers import SGD,RMSprop
from keras.layers import Conv2D, MaxPooling2D

import keras_resnet
from keras_resnet.models import ResNet50
from keras_resnet.models import ResNet18

from sklearn.neural_network import MLPClassifier

Data_reader = csv_reader.CsvReader('../DataSet')
Data_writer = csv_reader.CsvReader('../output')

if __name__ == '__main__':

    train_data = Data_reader.read_data('train.csv')
    test_data = Data_reader.read_data('test.csv')

    X_train = train_data.iloc[:, 1:]
    Y_train = train_data.iloc[:, 0]
    X_test = test_data

    # train_data_bin = Binarizer(threshold=127).fit_transform(train_data)
    # X_test_bin = Binarizer(threshold=127).fit_transform(test_data)
    # X_train = pd.DataFrame(train_data_bin).iloc[:, 1:]
    # Y_train = train_data.iloc[:, 0]
    # X_test = pd.DataFrame(X_test_bin)

    train_images, vali_images, train_labels, vali_labels = \
        train_test_split(X_train, Y_train, train_size=0.99,random_state=1)
    print('start predict')

    train_labels_categorical = keras.utils.to_categorical(train_labels, num_classes=10)
    vali_labels_categorical = keras.utils.to_categorical(vali_labels, num_classes=10)

    train_images_2d = train_images.values.reshape(train_images.shape[0],28,28,1)
    vali_images_2d = vali_images.values.reshape(vali_images.shape[0],28,28,1)

    test_images_2d = X_test.values.reshape(X_test.shape[0],28,28,1)

    shape, classes = (28, 28, 1), 10

    x = keras.layers.Input(shape)
    model = ResNet18(x, classes=classes)

    RMS = RMSprop(lr=0.0001)
    model.compile("adam", "categorical_crossentropy", ["accuracy"])
    # model.compile(optimizer=RMS, loss="binary_crossentropy",metrics=['accuracy'])

    # model = Sequential()
    # model.add(Conv2D(32, (3, 3),padding='same', activation='relu', input_shape=(28, 28, 1)))
    # model.add(Conv2D(32, (3, 3),padding='same', activation='relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Dropout(0.25))
    #
    # model.add(Conv2D(64, (3, 3),padding='same', activation='relu'))
    # model.add(Conv2D(64, (3, 3),padding='same', activation='relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Dropout(0.25))
    #
    # model.add(Flatten())
    # model.add(Dense(512, activation='relu'))
    # model.add(Dropout(0.5))
    # model.add(Dense(256, activation='relu'))
    # model.add(Dropout(0.5))
    # model.add(Dense(10, activation='softmax'))

    # sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
    # RMS= RMSprop(lr=0.0001)
    # model.compile(loss='binary_crossentropy', optimizer=RMS)

    model.fit(train_images_2d, train_labels_categorical,epochs=50,batch_size=128)
    score = model.evaluate(vali_images_2d, vali_labels_categorical, batch_size=128)

    results = model.predict(test_images_2d)

    res_data = np.argmax(results, axis=1)

    Data_writer.write_data_without_index(res_data, 'submissionRES.csv', columns=('Label',),index_name='ImageId')
