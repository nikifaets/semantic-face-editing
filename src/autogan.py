from autoencoder import Autoencoder 
import tensorflow as tf 
from tensorflow import keras
from tensorflow.keras import layers, models
import numpy as np 
from dataset_utils import shuffle_with_labels
import visualize

class AutoGAN:

    def __init__(self, width, height, train=None, test=None):

        self.width = width
        self.height = height
        if train is not None:
            (self.positive, self.negative) = train

        if test is not None:
            (self.positive_test, self.negative_test) = test

        self.kernel_size = (3,3)
        self.model = None

        self.model_created = False

        self.ae = Autoencoder(width, height, train, test)
        self.d_net = None
        print("gan initiated")

    def discriminator(self, input):

        conv1 = layers.Conv2D(64, (3,3), activation='relu', padding='same', name='discriminator_input')(input)
        pool1 = layers.MaxPooling2D(2,2)(conv1)
        conv2 = layers.Conv2D(32, (3,3), activation='relu', padding='same')(pool1)
        pool2 = layers.MaxPooling2D(2,2)(conv2)
        conv3 = layers.Conv2D(16, (3,3), activation='relu', padding='same')(pool2)
        pool3 = layers.MaxPooling2D(2,2)(conv3)
        conv4 = layers.Conv2D(4, (3,3), activation='relu', padding='same')(pool3)
        flatten = layers.Flatten()(conv4)
        output = layers.Dense(1, activation='sigmoid', name='discriminator_output')(flatten)

        print("discriminator layers created")
        return output

    
    def create_model(self):

        self.ae.create_model()
        self.ae_model = self.ae.get_model()
        self.ae_model.summary()

        d_input_layer = tf.keras.Input(shape=(self.width, self.height, 3))


        ae_net = self.ae.get_net()
        
        self.d_net = self.discriminator(d_input_layer)
        
        #compile discriminator model
        self.d_model = keras.Model(inputs = d_input_layer, outputs = self.d_net, name="discriminator")
        self.d_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        #compile autogan model
        self.d_model.trainable = False
        self.autogan_model = tf.keras.Sequential()
        self.autogan_model.add(self.ae_model)
        self.autogan_model.add(self.d_model)
        self.autogan_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        self.autogan_model.summary()
        print("autogan created")




        '''def autogan_loss(self,y_true, y_pred):
            
            ae_input = self.ae_model.input
            ae_output = self.ae_model.output

            ae = keras.Model(inputs = ae_input, outputs = ae_output)
            ae.predict(y_)'''
            
    def train(self, epochs, batch_size):

        
        batches_per_epoch = self.negative.shape[0] // batch_size

        for epoch_num in range(0, epochs):

            shuffle_with_labels(self.negative, 0)
            shuffle_with_labels(self.positive, 1)

            for batch in range(0, batches_per_epoch):

                curr_batch_start = batch * batch_size

                negative_batch = self.negative[curr_batch_start : (curr_batch_start + batch_size)]

                loss = 0.0

                # ! epoch%2 train autogan 
                if epoch_num % 5 == 0:
                    
                    print("train autogan")
                    labels = np.ones(shape=batch_size)
                    loss, _ = self.autogan_model.train_on_batch(negative_batch, labels)
                # epoch%2 train discriminator
                
                else:
                    
                    print("train discriminator")
                    positive_batch = self.positive[curr_batch_start : (curr_batch_start + batch_size)]
                    mixed_batch = np.concatenate((negative_batch, positive_batch))
                    labels = np.concatenate((np.zeros(shape=(batch_size)), np.ones(shape=(batch_size))))

                    loss, _ = self.d_model.train_on_batch(mixed_batch, labels)



                print("epoch: %d, batch: %d loss: %f" % (epoch_num, batch, loss))

                if epoch_num %3 == 0:

                    self.autogan_model.save_weights(filepath="../checkpoints_autogan/epoch" + str(epoch_num))
                    #print("VALIDATE DISCRIMINATOR")
                    #visualize.test_discriminator("../checkpoints_autogan/epoch" + str(epoch_num), self.negative_test, self.positive_test)
        
    def get_discriminator(self):

        return self.d_net

    def get_discriminator_model(self):

        return self.d_model

    def get_autogan_model(self):

        if not self.model_created:
            self.create_model()
        return self.autogan_model