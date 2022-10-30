from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Convolution2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten

def conv():
    modelo=Sequential()
    modelo.add(Convolution2D(32,(3,3),input_shape=(224,224,3),activation='relu'))
    modelo.add(MaxPooling2D(pool_size=((2,2))))
    modelo.add(Flatten())
    return modelo
