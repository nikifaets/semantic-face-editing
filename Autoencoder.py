import tensorflow as tf 
from tensorflow import keras
from tensorflow.keras import layers, models


class Autoencoder:

	def __init__(self, width, height, x_train, y_train, x_test, y_test):

		self.width = width
		self.height = height
		self.x_train = y_train
		self.y_train = y_train
		self.x_test = x_test
		self.y_test = y_test

	def encode(self, input):

		conv1 = layers.Conv2D(16, (2,2), activation='relu', padding='same')(input)
		pool1 = layers.MaxPooling2D((2,2))(conv1)

		return pool1

	def decode(self, latent_space):

		conv_transpose1 = layers.UpSampling2D(size=(2,2))(latent_space)
		conv1 = layers.Conv2D(16, (2,2), activation='relu', padding='same')(conv_transpose1)

		return conv1


	def create_model(self):

		input_layer = tf.keras.Input(shape=(self.width, self.height, 1))
		latent_space = self.encode(input_layer)
		output_layer = self.decode(latent_space)

		self.model = keras.Model(inputs=input_layer, outputs=output_layer, name="model")
		self.model.summary()

	def train_model(self, _batch_size, _epochs):

		self.model.compile(loss=keras.losses.MeanSquaredError())
		self.model.fit(self.x_train, self.y_train, batch_size=_batch_size, epochs=_epochs, verbose=1)

	def get_model(self):

		return self.model

	def save_model(self, filename):

		self.model.save_weights(filename)


	def load_model(self, filename):

		self.model.load_weights(filename)

