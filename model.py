import keras
from keras.layers import Input, Conv2D, Dense, MaxPooling2D, Flatten, Conv2DTranspose, merge, Dropout, UpSampling2D
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

def ship_cnn(input_dim=(512, 512, 3)):
	inputs = Input(shape=input_dim)

	conv1 = Conv2D(filters=64, kernel_size=3, padding='same', kernel_initializer='he_normal')(inputs)
	pool1 = MaxPooling2D(pool_size=(2, 2), padding='same')(conv1)

	conv2 = Conv2D(filters=128, kernel_size=3, padding='same', kernel_initializer='he_normal')(pool1)
	drop2 = Dropout(0.5)(conv2)
	pool2 = MaxPooling2D(pool_size=(2, 2), padding='same')(drop2)

	conv3 = Conv2D(filters=256, kernel_size=3, padding='same', kernel_initializer='he_normal')(pool2)
	drop3 = Dropout(0.5)(conv3)

	# Up-L1 - merge upsampled L2 with output from L1
	#upsampled_conv1 = Conv2DTranspose(filters=32, kernel_size=3, strides=2, padding='same')(conv3)
	up1 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(drop3))
	merge1 = merge([drop2, up1], mode='concat', concat_axis=3)
	conv4 = Conv2D(filters = 128, kernel_size=3, padding='same')(merge1)

	# Up-L1 - merge upsampled L2 with output from L1
	#upsampled_conv2 = Conv2DTranspose(filters=64, kernel_size=3, strides=2, padding='same')
	up2 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv4))
	merge2 = merge([conv1, up2], mode='concat', concat_axis=3)
	conv5 = Conv2D(filters = 64, kernel_size=3, padding='same', kernel_initializer='he_normal')(merge2)

	conv6 = Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
	# Final layer - makes 768x768x1 image
	segmap = Conv2D(filters = 1, kernel_size = 1, activation = 'sigmoid')(conv6)

	model = Model(inputs=inputs, outputs=segmap)
	print(model.summary())
	model.compile(optimizer = Adam(lr = 0.0001), loss = 'binary_crossentropy', metrics = ['accuracy'])
	return model


def test_unet(input_dim=(768, 768, 3)):
	inputs = Input(input_dim)
	conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
	conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
	pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

	conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
	conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
	pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

	conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
	conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
	pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

	conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
	conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
	drop4 = Dropout(0.5)(conv4)
	pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

	conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
	conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
	drop5 = Dropout(0.5)(conv5)

	up6 = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(drop5))
	merge6 = merge([drop4, up6], mode='concat', concat_axis=3)
	conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
	conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)

	up7 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv6))
	merge7 = merge([conv3, up7], mode='concat', concat_axis=3)
	conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
	conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)

	up8 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv7))
	merge8 = merge([conv2, up8], mode='concat', concat_axis=3)
	conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
	conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)

	up9 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv8))
	merge9 = merge([conv1, up9], mode='concat', concat_axis=3)
	conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
	conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
	conv9 = Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
	conv10 = Conv2D(1, 1, activation='sigmoid')(conv9)

	model = Model(input=inputs, output=conv10)

	model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy'])

	return model
