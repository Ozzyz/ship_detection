import keras
from keras.layers import Input, Conv2D, Dense, MaxPooling2D, Flatten, Conv2DTranspose
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
    return model


def hourglass_cnn(input_dim=(512, 512, 3)):
    inputs = Input(shape=input_dim)

    conv1 = Conv2D(filters=32, kernel_size=3)(inputs)
    pool1 = MaxPooling2D(pool_size=(2,2))(conv1)
    
    conv2 = Conv2D(filters=64, kernel_size=3)(pool1)
    pool2 = MaxPooling2D(pool_size=(2,2))(conv2)

    #conv3 = Conv2D(filters=128, kernel_size=3)(pool2)
    #pool3 = MaxPooling2D(pool_size=(2,2))(conv3)

   # conv4 = Conv2D(filters=256, kernel_size=3)(pool3)
   # pool4 = MaxPooling2D(pool_size=(2,2))(conv4)

   # conv5 = Conv2D(filters=512, kernel_size=3)(pool4)

   # deconv1 = Conv2DTranspose(filters=512, kernel_size=3)(conv5)

   # deconv2 = Conv2DTranspose(filters=256, kernel_size=3)(deconv1)

    #deconv3 = Conv2DTranspose(filters=128, kernel_size=3)(pool3)

    deconv4 = Conv2DTranspose(filters=64, kernel_size=3)(pool2)

    deconv5 = Conv2DTranspose(filters=32, kernel_size=3)(deconv4)
    flattened = Flatten()(deconv5)
    dense = Dense(256, activation ='relu')(flattened)

    predictions = Dense(10, activation='softmax')(dense)

    model = Model(inputs=inputs, outputs=predictions)
    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model
