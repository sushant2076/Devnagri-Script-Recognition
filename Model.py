import pandas as pd
import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt
import pickle
import sklearn
import cv2
import os
from keras.layers import Dense, Flatten, Conv2D, MaxPool2D


X = pickle.load(open('X_train.pickle','rb'))
y = pickle.load(open('y_train.pickle','rb'))

model  = keras.Sequential([
    Conv2D(64, (3,3), activation='relu'),
    MaxPool2D((2,2)),

    Conv2D(64, (3,3), activation='relu'),
    MaxPool2D((2,2)),

    Conv2D(64, (3,3), activation='relu'),
    MaxPool2D((2,2)),

    Conv2D(64, (3,3), activation='relu'),
    MaxPool2D((2,2)),

    Flatten(),

    Dense(512, input_shape=(220, 220), activation='relu'),

    Dense(48, activation='softmax')
])

model.compile(loss = 'sparse_categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

model.fit(X, y, batch_size=100, epochs = 3)

model.save('model5.h5')


