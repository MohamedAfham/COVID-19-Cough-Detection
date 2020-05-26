import tensorflow.compat.v1 as tf
from tensorflow.compat.v1 import keras
from tensorflow.compat.v1.keras import Sequential,Model
from tensorflow.compat.v1.keras.layers import Dense, Dropout, Flatten,BatchNormalization
from tensorflow.compat.v1.keras.layers import Conv2D, MaxPooling2D, Activation,ReLU,LeakyReLU

def neural_network(input_shape):
    inputs = keras.Input(shape=input_shape)

    #Layer 1 
    x = MaxPooling2D(pool_size=(2,2),name='MaxPooling2D_1')(inputs)
    x = Conv2D(32, kernel_size=(5,5),padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = MaxPooling2D(pool_size=(4,4))(x)

    #Layer 2
    x = Conv2D(64, kernel_size=(5,5),padding='same',name='Conv2D_2')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = MaxPooling2D(pool_size=(2,2),name='MaxPooling2D_3')(x)

    x = Flatten(name = 'Flatten')(x)

    #Layer 3
    #model.add(Dense(256,name = 'Dense_1'))
    #model.add(BatchNormalization(name = 'BatchNormalization_2'))
    #model.add(LeakyReLU(alpha=0.1))
    #model.add(Dropout(0.5,name = 'Dropout_1'))

    #Layer 4
    x = Dense(128,name = 'Dense_2')(x)
    x = BatchNormalization(name = 'BatchNormalization_3')(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = Dropout(0.5,name = 'Dropout_2')(x)

    #Layer 5
    x = Dense(128,name = 'Dense_3')(x)
    x = BatchNormalization(name = 'BatchNormalization_4')(x)
    x = LeakyReLU(alpha=0.1)(x)
    #model.add(Dropout(0.5,name = 'Dropout_3'))

    outputs = Dense(1,activation='sigmoid',name = 'Dense_4')(x)

    model = Model(inputs,outputs)
    return model