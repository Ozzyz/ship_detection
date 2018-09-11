import keras
from keras.layers import Input, Conv2D, Dense, MaxPooling2D, Flatten, Conv2DTranspose
from keras.layers.merge import concatenate
from keras.models import Model
from keras.optimizers import Adam
from keras import backend as K
from loss_funcs import dice_coef, true_positive_rate, dice_p_bce

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

	conv3 = Conv2D(filters=128, kernel_size=3)(pool2)
	pool3 = MaxPooling2D(pool_size=(2,2))(conv3)

	conv4 = Conv2D(filters=256, kernel_size=3)(pool3)
	pool4 = MaxPooling2D(pool_size=(2,2))(conv4)

	conv5 = Conv2D(filters=512, kernel_size=3)(pool4)

	deconv1 = Conv2DTranspose(filters=512, kernel_size=3)(conv5)

	deconv2 = Conv2DTranspose(filters=256, kernel_size=3)(deconv1)

	deconv3 = Conv2DTranspose(filters=128, kernel_size=3)(pool3)

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
	conv1 = Conv2D(filters = 64, kernel_size = 3, padding='same')(inputs)
	conv2 = Conv2D(filters = 64, kernel_size = 3, padding='same')(conv1)
	# L2
	down1 = MaxPooling2D(pool_size=(2,2), padding='same')(conv2)
	conv3 = Conv2D(filters = 128, kernel_size = 3, padding='same')(down1)
	conv4 = Conv2D(filters = 128, kernel_size = 3, padding='same')(conv3)
	# L3
	down2 = MaxPooling2D(pool_size=(2,2), padding='same')(conv4)
	conv5 = Conv2D(filters = 256, kernel_size = 3, padding='same')(down2)
	conv6 = Conv2D(filters = 256, kernel_size = 3, padding='same')(conv5)
	# L4
	down3 = MaxPooling2D(pool_size=(2,2), padding='same')(conv6)
	conv7 = Conv2D(filters = 512, kernel_size = 3, padding='same')(down3)
	conv8 = Conv2D(filters = 512, kernel_size = 3, padding='same')(conv7)
	# L5
	down4 = MaxPooling2D(pool_size=(2,2), padding='same')(conv8)
	conv9 = Conv2D(filters=1024, kernel_size = 3, padding='same')(down4)
	conv10 = Conv2D(filters=1024, kernel_size=3, padding='same')(conv9)

	# Merge upsampled L5 with output from L4
	upsampled_conv10 = Conv2DTranspose(filters=512, kernel_size=3, strides=2, padding='same')(conv10)
	print(upsampled_conv10.shape)
	# Up-L4
	up1 = concatenate([conv8, upsampled_conv10])
	conv11 = Conv2D(filters = 512, kernel_size= 3, padding='same')(up1)
	conv12 = Conv2D(filters = 512, kernel_size= 3, padding='same')(conv11)
	# Up-L3 - merge upsampled L4 with output from L3
	upsampled_conv12 = Conv2DTranspose(filters=256, kernel_size = 3, strides=2, padding='same')(conv12)
	up2 = concatenate([conv6, upsampled_conv12])
	conv13 = Conv2D(filters = 256, kernel_size= 3, padding='same')(up2)
	conv14 = Conv2D(filters = 256, kernel_size= 3, padding='same')(conv13)
	# Up-L2 - merge upsampled L3 with output from L2
	upsampled_conv14 = Conv2DTranspose(filters=128, kernel_size = 3, strides=2, padding='same')(conv14)
	up3 = concatenate([conv4, upsampled_conv14])
	conv15 = Conv2D(filters = 128, kernel_size= 3, padding='same')(up3)
	conv16 = Conv2D(filters = 128, kernel_size= 3, padding='same')(conv15)
	# Up-L1 - merge upsampled L2 with output from L1
	upsampled_conv16 =  Conv2DTranspose(filters=64, kernel_size = 3, strides=2, padding='same')(conv16)
	up4 = concatenate([conv2, upsampled_conv16])
	conv17 = Conv2D(filters = 64, kernel_size = 3, padding='same')(up4)
	conv18 = Conv2D(filters = 64, kernel_size= 3, padding='same')(conv17)

	conv19 = Conv2D(filters = 2, kernel_size = 3, padding='same')(conv18)
	# Final layer - makes 768x768x1 image
	segmap = Conv2D(filters = 1, kernel_size = 1, activation = 'sigmoid')(conv19)

	model = Model(inputs=inputs, outputs=segmap)
	print(model.summary())
	model.compile(optimizer = Adam(1e-4, decay=1e-6), loss = dice_p_bce, metrics = ['accuracy', dice_coef, true_positive_rate])
	return model
