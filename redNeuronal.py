from tensorflow.keras.layers import Dense
from convolucion import *

def neuronal():
    modelo=conv()
    modelo.add(Dense(128,activation='relu'))
    modelo.add(Dense(50,activation='relu'))
    modelo.add(Dense(1,activation='sigmoid'))
    modelo.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
    return modelo

