
from model import basic_cnn, hourglass_cnn
from keras.datasets import mnist
from keras import utils

(x_train, y_train), (x_test, y_test) = mnist.load_data()

num_classes = 10
img_rows, img_cols = (28, 28)

x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
input_shape = (img_rows, img_cols, 1)


x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

# convert class vectors to binary class matrices
y_train = utils.to_categorical(y_train, num_classes)
y_test = utils.to_categorical(y_test, num_classes)


#model = basic_cnn(input_dim=(28, 28, 1))
model = hourglass_cnn(input_dim=(28, 28, 1))

print(model.summary())
model.fit(x_train, y_train,
          validation_data=(x_test, y_test))
