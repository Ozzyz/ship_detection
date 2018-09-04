import keras
from keras.layers import Input, Conv2D, Dense, MaxPooling2D, Flatten
from keras.models import Model


def basic_cnn(input_dim=(512, 512, 3)):
    inputs = Input(shape=input_dim)

    conv1 = Conv2D(filters=64, kernel_size=3)(inputs)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(filters=32, kernel_size=3)(pool1)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    flat = Flatten()(pool2)
    fully_connected = Dense(256, activation='relu')(flat)
    predictions = Dense(10, activation='softmax')(fully_connected)

    model = Model(inputs=inputs, outputs=predictions)
    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    print(model.summary())
    return model


m = basic_cnn()
