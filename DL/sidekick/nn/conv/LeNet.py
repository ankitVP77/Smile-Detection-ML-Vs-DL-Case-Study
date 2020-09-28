from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Activation, MaxPool2D, Flatten, Dense
import tensorflow.keras.backend as K
from tensorflow import nn as tfn

class LeNet:
    @staticmethod
    def build(width,height,depth,classes):
        model=Sequential()
        inputShape=(height,width,depth)
        if K.image_data_format()=='channels_first':
            inputShape=(depth,height,width)

        model.add(Conv2D(16,(3,3),padding='same',input_shape=inputShape))
        model.add(Activation(tfn.relu))
        model.add(MaxPool2D((2,2),(2,2)))

        model.add(Conv2D(32,(5,5),padding='same'))
        model.add(Activation(tfn.relu))
        model.add(MaxPool2D((2,2),(2,2)))

        model.add(Flatten())
        model.add(Dense(500))
        model.add(Activation(tfn.relu))
        model.add(Dense(classes))
        model.add(Activation(tfn.softmax))

        return model