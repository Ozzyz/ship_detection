import keras
from keras.layers import Input, Conv2D, Dense
from keras.models import Model


def basic_cnn(input_dim=(512, 512, 3)):
    inputs = Input(shape=input_dim)

    conv1 = Conv2D(filters=64, kernel_size=3)(inputs)
    conv2 = Conv2D(filters=32, kernel_size=3)(conv1)

    fully_connected = Dense(256, activation='relu')(conv2)
    predictions = Dense(10, activation='softmax')(fully_connected)

    model = Model(inputs=inputs, outputs=predictions)
    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


m = basic_cnn()
