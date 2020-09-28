from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Activation, BatchNormalization, MaxPool2D, Dropout, Dense, Flatten
from tensorflow import nn as tfn
import tensorflow.keras.backend as K

class ReLeNet:
    @staticmethod
    def build(width, height, depth, classes):
        inputShape= (height, width, depth)
        chanDim= -1
        if K.image_data_format()=="channels_first":
            inputShape= (depth, height, width)
            chanDim= 1

        model= Sequential()
        model.add(Conv2D(16, (3,3), padding='same', input_shape=inputShape))
        model.add(Activation(tfn.relu))
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
        model.add(Dropout(0.25))

        model.add(Conv2D(32, (5, 5), padding='same'))
        model.add(Activation(tfn.relu))
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
        model.add(Dropout(0.25))

        model.add(Flatten())
        model.add(Dense(512))
        model.add(Activation(tfn.relu))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))

        model.add(Dense(classes))
        model.add(Activation(tfn.softmax))

        return model