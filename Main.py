import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dense,Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

(X_train, y_train),(X_test, y_test)= keras.datasets.boston_housing.load_data(test_split=0.3)
X_train, x_valid, y_train,y_valid= train_test_split(X_train, y_train,test_size=0.2)
print("X")
print(X_train)
print("Y")
print(y_train)
print("X shape")
print(X_train.shape)
print("y shape")
print(y_train.shape)
mms = MinMaxScaler()
X_train = mms.fit_transform(X_train)
x_valid = mms.transform(x_valid)
X_test = mms.transform(X_test)


def build_model():
    model = keras.Sequential()
    model.add(Dense(13))
    model.add(Dense(254,activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(128,activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mae',metrics=['mse'])
    return model
    
model = build_model() 
model.fit(X_train, y_train,epochs = 150,validation_data = (x_valid,y_valid),batch_size=16)
print(model.evaluate(X_test,y_test))
model.summary()
    

