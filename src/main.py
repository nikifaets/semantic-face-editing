from Autoencoder import Autoencoder
import tensorflow_datasets as tfds
from tensorflow.keras import datasets
import numpy as np

(x_train,  y_train), (x_test, y_test) = datasets.mnist.load_data()

print("type ", type(x_train))
x_train = x_train / 255.0

x_train = x_train[:100]

x_train = np.reshape(x_train, (x_train.shape[0],28,28,1))
print(x_train.shape)
ae = Autoencoder(28, 28, x_train, x_train, x_test, x_test)
ae.create_model()
#ae.train_model(30, 1)
ae.load_model("hehehe.h5")

model = ae.get_model()
print(x_train[0].shape)
res = model.predict(x_train[:2])

print(res)


