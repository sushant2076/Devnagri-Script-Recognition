import pandas as pd
import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt
import pickle
import sklearn
import cv2
import os
from keras.layers import Dense, Flatten, Conv2D, MaxPool2D

X_test = pickle.load(open('X_test.pickle','rb'))
y_test = pickle.load(open('y_test.pickle','rb'))


model = keras.models.load_model('model5.h5')

model.evaluate(X_test, y_test)

#model1 = accuracy: 0.6905 e=5
#model2 = accuracy : 0.7672 e=2 
#model3 = accuracy: 0.6926 e=2 
#model3 = accuracy: 0.7500 e=5 
#model4 = accuracy: 0.7482 e=3 
#model5 = accuracy: 0.8949
