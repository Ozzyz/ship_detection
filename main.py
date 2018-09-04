
from model import basic_cnn
from keras.datasets import mnist


(x_train, y_train), (x_test, y_test) = mnist.load_data()


model = basic_cnn(input_dim=(28, 28, 1))

model.fit(x_train, y_train)

a = model.evaluate(x_test, y_test)
print(a)
