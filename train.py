#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 2019
@author: jiazx
"""
import tensorflow as tf
import keras
from keras.layers import Dense,Flatten,Dropout
from keras.utils.np_utils import to_categorical
from keras.utils import plot_model
from keras.models import Sequential
from keras.callbacks import TensorBoard
import csv
import numpy as np

# read and process data from csv file
def read_data(data_path):
    with open(data_path) as fr:
        reader = csv.reader(fr)
        headerRow = next(reader)
        n = len(headerRow)
        listFeats = []; labels = []
        for row in reader:
            fltRow = list(map(float,row))
            listFeats.append(fltRow[0:n-1])
            labels.append(int(fltRow[-1]))

    # convert list to numpy array
    listFeats = np.array(listFeats)
    labels = np.array(labels)
    labels = to_categorical(labels,2)

    # split dataset
    x_train = listFeats/200.0
    y_train = labels
    x_test = listFeats[0:53]/200.0
    y_test = labels[0:53]

    return x_train,y_train, x_test, y_test

def train():
    # get training data & test data
    x_train,y_train, x_test,y_test = read_data(r'D:\python_ws\heartdisease\heart.csv')
    print(x_train.shape,y_train.shape,x_test.shape,y_test.shape)
    print(x_train[0],y_train[0])
    # input()
    # build a sequential model
    model = Sequential()
    model.add(Dense(input_dim=13,units=40,activation='relu'))
    model.add(Dense(30,activation='relu'))
    model.add(Dense(20,activation='relu'))
    model.add(Dense(2,activation='softmax'))
    # use adam optimizer
    op = keras.optimizers.SGD(lr=0.01)
    model.compile(loss='categorical_crossentropy',optimizer=op,metrics=['accuracy'])
    # print the model structure
    model.summary()
    # draw
    plot_model(model)

    tb = TensorBoard(log_dir=r'D:\python_ws\heartdisease')

    # start training
    model.fit(x_train,y_train,batch_size=8,epochs=200,callbacks=[tb])
    # save the model so that you can directly use next time
    model.save(r'D:\python_ws\heartdisease\model.h5')

    # evaluate the trained model
    test_score = model.evaluate(x_test,y_test)
    print('test accuracy: ',test_score[1])

    # predict 
    result = model.predict(x_test[0:2])
    print(result)
if __name__ == "__main__":
    train()
