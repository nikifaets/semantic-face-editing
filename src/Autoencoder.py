import tensorflow as tf 
from tensorflow import keras
from tensorflow.keras import layers, models


class Autoencoder:

	def __init__(self, width, height, train=None, test=None, training=True):

		self.width = width
		self.height = height
		self.x_train = train
		self.y_train = train
		self.x_test = test
		self.y_test = test
		self.kernel_size = (3,3)
		self.model = None
		self.net = None
		self.model_created = False
		self.training = training

	def encode(self, input):

		conv1 = layers.Conv2D(64, (3,3), activation='relu', padding='same', name="autoencoder_input")(input)
		pool1 = layers.MaxPooling2D((2,2))(conv1)
		conv2 = layers.Conv2D(64, (3,3), activation='relu', padding='same')(pool1)
		pool2 = layers.MaxPooling2D(2,2)(conv2)
		conv3 = layers.Conv2D(32, (3,3), activation='relu', padding='same')(pool2)
		#pool3 = layers.MaxPooling2D((2,2))(conv3)
		conv4 = layers.Conv2D(16, (3,3), activation='relu', padding='same')(conv3)
		conv5 = layers.Conv2D(12, (3,3), activation='relu', padding='same', name="latent-space")(conv4)
		return conv5
	
	def decode(self, latent_space, training):

		conv1 = layers.Conv2D(16, self.kernel_size, padding='same')(latent_space)
		bn1 = layers.BatchNormalization(-1)(conv1)
		relu1 = layers.ReLU()(bn1)
		dropout1 = layers.Dropout(rate=0.2)(relu1, training=training)

		conv2 = layers.Conv2D(32, self.kernel_size, padding='same')(dropout1)
		bn2 = layers.BatchNormalization(-1)(conv2)
		relu2 = layers.ReLU()(bn2)
		dropout2 = layers.Dropout(rate=0.2)(relu2, training=training)

		upsampling2 = layers.UpSampling2D((2,2))(dropout2)

		
		conv3 = layers.Conv2D(64, self.kernel_size, padding='same')(upsampling2)
		bn3 = layers.BatchNormalization(-1)(conv3)
		relu3 = layers.ReLU()(bn3)
		dropout3 = layers.Dropout(rate=0.2)(relu3, training=training)

		upsampling3 = layers.UpSampling2D((2,2))(dropout3)

		conv4 = layers.Conv2D(64, self.kernel_size, activation='relu', padding='same')(upsampling3)

		output = layers.Conv2D(3, (2,2), activation='tanh', padding='same', name='autoencoder_output')(conv4)
		#output = layers.Flatten()(output)
		#output = layers.Dense(1)(output)
		return output


	def create_model(self):

		self.model_created = True

		input_layer = tf.keras.Input(shape=(self.width, self.height, 3))
		latent_space = self.encode(input_layer)
		output_layer = self.decode(latent_space, self.training)
		self.net = output_layer
		self.model = keras.Model(inputs=input_layer, outputs=output_layer, name="autoencoder")
	

	def train_model(self, _batch_size, _epochs, _verbose):

		checkpoint_filepath = "../checkpoints/weights.{epoch:02d}"
		model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
			filepath=checkpoint_filepath,
			save_weights_only=True,
			monitor='loss',
			mode='min',
			save_best_only=True)

		self.model.compile(loss=keras.losses.MeanSquaredError(), optimizer = 'adam')
		self.model.fit(self.x_train, self.y_train, batch_size=_batch_size, epochs=_epochs, verbose=_verbose, callbacks=[model_checkpoint_callback])#, steps_per_epoch=1, validation_steps=1)

	def get_model(self):

		return self.model

	def save_model(self, filename):

		self.model.save_weights(filename)

	def get_net(self):

		return self.net

	def load_model(self, filename):

		if not self.model_created:
			self.create_model()

		self.model.load_weights(filename)

