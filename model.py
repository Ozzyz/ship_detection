import keras
from keras.layers import Input, Conv2D, Dense, MaxPooling2D, Flatten, Conv2DTranspose, merge
from keras.models import Model
from keras.optimizers import Adam

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


def hourglass_cnn(input_dim=(512, 512, 1)):
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


def unet(input_dim = (512,512,3)):
    inputs = Input(shape=input_dim)
    # L1
    conv1 = Conv2D(filters = 64, kernel_size = 3)(inputs)
    conv2 = Conv2D(filters = 64, kernel_size = 3)(conv1)
    # L2
    down1 = MaxPooling2D(pool_size=(2,2))(conv2)
    conv3 = Conv2D(filters = 128, kernel_size = 3)(down1)
    conv4 = Conv2D(filters = 128, kernel_size = 3)(conv3)
    # L3
    down2 = MaxPooling2D(pool_size=(2,2))(conv4)
    conv5 = Conv2D(filters = 256, kernel_size = 3)(down2)
    conv6 = Conv2D(filters = 256, kernel_size = 3)(conv5)
    # L4
    down3 = MaxPooling2D(pool_size=(2,2))(conv6)
    conv7 = Conv2D(filters = 512, kernel_size = 3)(down3)
    conv8 = Conv2D(filters = 512, kernel_size = 3)(conv7)
    # L5 
    down4 = MaxPooling2D(pool_size=(2,2))(conv8)
    conv9 = Conv2D(filters = 1024, kernel_size = 3)(down4)
    conv10 = Conv2D(filters = 1024, kernel_size=3)(conv9)

    # Merge upsampled L5 with output from L4
    upsampled_conv10 = Conv2DTranspose(filters=512, kernel_size = 3)(conv10)
    # Up-L4 
    up1 = merge([conv8, upsampled_conv10], mode = 'concat')
    conv11 = Conv2D(filters = 512, kernel_size= 3)(up1)
    conv12 = Conv2D(filters = 512, kernel_size= 3)(conv11)
    # Up-L3 - merge upsampled L4 with output from L3
    upsampled_conv12 = Conv2DTranspose(filters=256, kernel_size = 3)(conv12)
    up2 = merge([conv6, upsampled_conv12], mode = 'concat')
    conv13 = Conv2D(filters = 256, kernel_size= 3)(up2)
    conv14 = Conv2D(filters = 256, kernel_size= 3)(conv13)
    # Up-L2 - merge upsampled L3 with output from L2
    upsampled_conv14 = Conv2DTranspose(filters=128, kernel_size = 3)(conv14)
    up3 = merge([conv4, upsampled_conv14], mode = 'concat')
    conv15 = Conv2D(filters = 128, kernel_size= 3)(up3)
    conv16 = Conv2D(filters = 128, kernel_size= 3)(conv15)
    # Up-L1 - merge upsampled L2 with output from L1
    upsampled_conv16 =  Conv2DTranspose(filters=64, kernel_size = 3)(conv16)
    up4 = merge([conv2, upsampled_conv16], mode= 'concat')(upsampled_conv16)
    conv17 = Conv2D(filters = 64, kernel_size = 3)(up4)
    conv18 = Conv2D(filters = 64, kernel_size= 3)(conv17)
    
    final_conv = Conv2D(filters = 2, kernel_size = 1)(conv18)

    segmap = Conv2D(filters = 1, kernel_size = 1, activation = 'sigmoid')(final_conv)

    model = Model(inputs=inputs, outputs=segmap)
    
    model.compile(optimizer = Adam(lr = 0.0001), loss = 'binary_crossentropy', metrics = ['accuracy'])
    return model






        
    