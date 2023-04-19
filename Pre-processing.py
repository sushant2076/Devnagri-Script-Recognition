import pandas as pd
import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt
import pickle
import sklearn
import cv2
import os

rootdir = r'C:\Users\acer\OneDrive\Desktop\MLIP Project\DEVNAGARI_NEW\TEST'

Categories = os.listdir(rootdir)

data  = []

for category in Categories:
    letter = os.path.join(rootdir,category)
    label = int(category)-1
    for img in os.listdir(letter):
        img_path = os.path.join(letter,img)
        img_read = cv2.imread(img_path)
        img_read = cv2.resize(img_read, (220,220))
        data.append([img_read,label])

np.random.shuffle(data)

X_train  = []
y_train = []

for img, label in data:
    X_train.append(img)
    y_train.append(label)

X_train = np.array(X_train)
y_train = np.array(y_train)

X_train = X_train/255

#print(Categories)
#print(X.shape)
#train ['1', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '2', 
# '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '3', '30'
# , '31', '32', '33', '34', '35', '36', '37', '38', '39', '4', '40', '41', '42',
#  '43', '44', '45', '46', '47', '48', '5', '6', '7', '8', '9']
#(6528, 128, 128, 3)

pickle.dump(X_train, open('X_test.pickle', 'wb'))
pickle.dump(y_train, open('y_test.pickle','wb'))
