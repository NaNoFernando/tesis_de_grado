import cv2
import numpy as np
import glob
import os
from random import shuffle
from redNeuronal import *
from tensorflow import keras
dataTr=[]
X_train=[]
X_test=[]
Y_test=[]
Y_train=[]

def distribucion():
    global dataTr
    for filename in glob.glob(os.path.join('data/train/malignant','*.jpg')):
        dataTr.append([1,cv2.imread(filename)])
    for filename in glob.glob(os.path.join('data/train/benign','*.jpg')):
        dataTr.append([0,cv2.imread(filename)])
    shuffle(dataTr)
    return dataTr

def almacenamiento_train():
    global X_train,Y_train
    for i,j in dataTr:
        X_train.append(j)
        Y_train.append(i)
    X_train=np.array(X_train)
    Y_train=np.array(Y_train)
    return X_train,Y_train

def almacenamiento_test():
    global X_test, Y_test
    for filename in glob.glob(os.path.join('data/test/malignant','*.jpg')):
        X_test.append(cv2.imread(filename))
        Y_test.append(1)
    for filename in glob.glob(os.path.join('data/test/benign','*.jpg')):
        X_test.append(cv2.imread(filename))
        Y_test.append(0)
    X_test=np.array(X_test)
    Y_test=np.array(Y_test)
    return X_test,Y_test

def entrenamiento():
    global X_train,Y_train,X_test,Y_test
    distribucion()
    almacenamiento_train()
    almacenamiento_test()
    modelo=neuronal()
    modelo.fit(X_train,Y_train,batch_size=32,epochs=9,validation_data=(X_test,Y_test))
    modelo.save("entrenamiento/mi_modelo")
    return modelo




