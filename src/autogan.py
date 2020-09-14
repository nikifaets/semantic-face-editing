from autoencoder import Autoencoder 
import tensorflow as tf 
from tensorflow import keras
from tensorflow.keras import layers, models
import numpy as np 
from dataset_utils import shuffle_with_labels, shuffle, normalize_tanh, normalize_pos
import datetime


class AutoGAN:

    def __init__(self, width, height, train=None, test=None, training=True):

        self.width = width
        self.height = height

        self.training = training

        if train is not None:
            (self.positive, self.negative) = train

        if test is not None:
            (self.positive_test, self.negative_test) = test

        self.kernel_size = (3,3)
        self.model = None

        self.model_created = False

        self.ae = Autoencoder(width, height, train, test, training=self.training)
        self.d_net = None
        print("gan initiated")

    def discriminator(self, input, training=True):

        conv1 = layers.Conv2D(64, (3,3), padding='same', strides=self.kernel_size, name='discriminator_input')(input)
        bn1 = layers.BatchNormalization(-1)(conv1)
        leaky1 = layers.LeakyReLU()(bn1)
        dropout1 = layers.Dropout(rate=0.2)(leaky1, training=training)

        #pool1 = layers.MaxPooling2D(2,2)(leaky1)
        conv2 = layers.Conv2D(32, (3,3), padding='same', strides=self.kernel_size )(dropout1)
        bn2 = layers.BatchNormalization(-1)(conv2)
        leaky2 = layers.LeakyReLU()(bn2)
        dropout2 = layers.Dropout(rate=0.2)(leaky2, training=training)

        #pool2 = layers.MaxPooling2D(2,2)(leaky2)
        conv3 = layers.Conv2D(16, (3,3), padding='same', strides=self.kernel_size)(dropout2)
        bn3 = layers.BatchNormalization(-1)(conv3)
        leaky3 = layers.LeakyReLU()(bn3)
        dropout3 = layers.Dropout(rate=0.2)(leaky3, training=training)

        #pool3 = layers.MaxPooling2D(2,2)(leaky3)
        conv4 = layers.Conv2D(4, (3,3), padding='same', strides=self.kernel_size)(dropout3)
        bn4 = layers.BatchNormalization(-1)(conv4)
        leaky4 = layers.LeakyReLU()(bn4)
        dropout4 = layers.Dropout(rate=0.2)(leaky4, training=training)

        flatten = layers.Flatten()(dropout4)
        output = layers.Dense(1, name='discriminator_output')(flatten)
        output_activation = layers.LeakyReLU()(output)

        print("discriminator layers created")
        return output_activation

    
    def create_model(self):

        self.model_created = True

        self.ae.create_model()
        self.ae_model = self.ae.get_model()
        self.ae_model.summary()

        input_layer = tf.keras.Input(shape=(self.width, self.height, 3))
        
        self.d_net = self.discriminator(input_layer, training=self.training)

        #compile discriminator model
        self.d_model = keras.Model(inputs = input_layer, outputs = [self.d_net], name="discriminator")
        self.d_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        self.d_model.summary()

        #compile autogan model
        self.d_model.trainable = False

        self.ae_out = self.ae_model(input_layer)
        self.disc_out = self.d_model(self.ae_out)

        print("ae_out ", self.ae_out)
        print("disc_out", self.disc_out)

        self.autogan_model = keras.Model(inputs = input_layer, outputs=[self.ae_out, self.disc_out], name="autogan")

        optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002)
        self.autogan_model.compile(optimizer=optimizer, loss={"autoencoder": self.ae_loss, "discriminator": keras.losses.binary_crossentropy}, loss_weights=[5,1], metrics=['accuracy'])
        self.autogan_model.summary()
        print("autogan created")


    def autogan_loss(self, y_true, y_pred):
        
        ae_in, label = y_true
        ae_out, disc_out = y_pred

        mse = keras.losses.MSE(ae_in, ae_out)
        bce = keras.losses.binary_crossentropy(label, disc_out)

        print("Calculating autoGAN loss: MSE: %f, BCE: %f" % (mse, bce))
        return mse + bce

    def ae_loss(self, y_true, y_pred):

        loss = keras.losses.mse(y_true, y_pred)
        loss = tf.math.reduce_mean(loss)

        return loss
            
    def train(self, epochs, batch_size):

        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S") #copied from tensorflow documentation
        summary_writer = tf.summary.create_file_writer("../logs/" + current_time)

        batches_per_epoch = self.negative.shape[0] // batch_size

        loss = 0.0
        ae_loss = 0.0
        disc_loss = 0.0
        disc_acc = 0.0
        ae_acc = 0.0
        for epoch_num in range(0, epochs+1):

            shuffle(self.negative)
            shuffle(self.positive)

            if epoch_num == 100:
                self.autogan_model.compile(optimizer='adam', loss={"autoencoder": self.ae_loss, "discriminator": keras.losses.binary_crossentropy}, loss_weights=[5,1], metrics=['accuracy'])

            
            for batch in range(0, batches_per_epoch):

                curr_batch_start = batch * batch_size

                negative_batch = self.negative[curr_batch_start : (curr_batch_start + batch_size)]

                # every n epochs train whole net
                if epoch_num % 2 == 0:

                    print("train all weights")
                    labels = np.ones(shape=batch_size)

                    metrics = self.autogan_model.train_on_batch(negative_batch, {"autoencoder": negative_batch, "discriminator": labels}, return_dict=True)
                    ae_loss = metrics['autoencoder_loss']
                    disc_loss = metrics['discriminator_loss']
                    disc_acc = metrics['discriminator_accuracy']
                    ae_acc = metrics['autoencoder_accuracy']

                    print("autogan metrics ", metrics)
                    loss = metrics['loss']

                    
                    

                # else train discriminator
                else:

                    print("train discriminator only")

                    positive_batch = self.positive[curr_batch_start : (curr_batch_start + batch_size)]
                    
                    ae_pred_batch = self.ae_model.predict(negative_batch)
                    normalize_tanh(ae_pred_batch)

                    subbatch_size = batch_size // 3
                    positive_batch = positive_batch[:subbatch_size]
                    negative_batch = negative_batch[:subbatch_size]
                    ae_pred_batch = ae_pred_batch[:subbatch_size]

                    mixed_batch = np.concatenate((negative_batch, positive_batch, ae_pred_batch))
                    labels = np.concatenate((np.zeros(shape=(subbatch_size)), np.ones(shape=(subbatch_size)), np.zeros(shape=subbatch_size)))
                    shuffle_with_labels(mixed_batch, labels)

                    loss, accuracy = self.d_model.train_on_batch(mixed_batch, labels)

                    disc_loss = loss
                    disc_acc = accuracy      

                step = epoch_num*batches_per_epoch + batch

                with summary_writer.as_default():
                    tf.summary.scalar("accuracy/autoencoder", ae_acc, step=step)
                    tf.summary.scalar("loss/autoencoder", ae_loss, step=step)
                    tf.summary.scalar("accuracy/discriminator", disc_acc, step=step)
                    tf.summary.scalar("loss/discriminator", disc_loss, step=step)


                print("epoch: %d, batch: %d loss: %f" % (epoch_num, batch, loss))
            
            # with summary_writer.as_default():

            #     step = epoch_num*batches_per_epoch + batch
            #     tf.summary.scalar("discriminator accuracy", disc_acc, step=step)
            #     tf.summary.scalar("autoencoder accuracy", ae_acc, step=step)
            #     tf.summary.scalar("discriminator loss", disc_loss, step=step)
            #     tf.summary.scalar("autoencoder loss", ae_loss, step=step)
            if epoch_num %2 == 0:

                self.autogan_model.save_weights(filepath="../checkpoints_autogan/epoch" + str(epoch_num))
                with summary_writer.as_default():

                    inp = self.negative[0:20]
                    out = self.ae_model.predict(inp)
                    normalize_pos(inp)
                    normalize_pos(out)

                    tf.summary.image("input/ae", inp, step=epoch_num, max_outputs=20)
                    tf.summary.image("output/ae", out, step=epoch_num, max_outputs=20)
                #print("VALIDATE DISCRIMINATOR")
                #visualize.test_discriminator("../checkpoints_autogan/epoch" + str(epoch_num), self.negative_test, self.positive_test)
    
    def load_trained(self, filename):
        
        self.create_model()
        self.autogan_model.load_weights(filename)

    def get_discriminator(self):

        return self.d_net

    def get_discriminator_model(self):

        return self.d_model

    def get_autogan_model(self):

        if not self.model_created:
            self.create_model()
        return self.autogan_model