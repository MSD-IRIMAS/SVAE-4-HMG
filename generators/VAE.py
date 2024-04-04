import sys
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import random

from sklearn.preprocessing import LabelEncoder as LE
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.mixture import GaussianMixture as GM

from utils.custom_callback import CustomReduceLRoP

#gpus = tf.config.list_physical_devices('GPU')
#for gpu in gpus:
#    tf.config.experimental.set_memory_growth(gpu, True)

class VAE:

    def __init__(self, output_directory, length_TS, n_joints,
                       dim, n_classes, batch_size=32, n_epochs=2000,
                       n_filters=128, kernel_size=40, activation='relu',
                       bottleneck_size=16, use_weighted_loss=True,
                       w_rec=0.999, w_kl=0.001,
                       verbose_encoder=False, verbose_decoder=False,
                       verbose_training=True):
        
        self.output_directory = output_directory

        self.length_TS = length_TS
        self.n_joints = n_joints
        self.dim = dim
        self.n_classes = n_classes

        self.n_filters = n_filters
        self.kernel_size = kernel_size
        self.batch_size = batch_size
        self.n_epochs = n_epochs

        self.activation = activation
        self.bottleneck_size = bottleneck_size

        self.use_weighted_loss = use_weighted_loss
        self.w_rec = w_rec
        self.w_kl = w_kl

        if not self.use_weighted_loss:

            self.w_rec, self.kl = 1.0, 1.0

        self.verbose_encoder = verbose_encoder
        self.verbose_decoder = verbose_decoder
        self.verbose_training = verbose_training

        self.build_encoder()
        if self.verbose_encoder:
            self.encoder.summary()

        self.build_decoder()
        if self.verbose_decoder:
            self.decoder.summary()

        self.optimizer = tf.keras.optimizers.Adam()

        self.reduce_lr = CustomReduceLRoP(
            factor=0.5, patience=50, min_lr=1e-4,
            optim_lr=self.optimizer.learning_rate)

    def sampling(self, mu_var):

        mu, var = mu_var

        epsilon = tf.keras.backend.random_normal(
            shape=tf.keras.backend.shape(mu), mean=0.0, stddev=1.0)
        random_sample = mu + tf.keras.backend.exp(var / 2) * epsilon

        return random_sample
    
    def conv_block(self, input_tensor, n_filters, kernel_size, activation='relu', strides=1, padding='same'):

        x = tf.keras.layers.Conv1D(
            filters=n_filters, kernel_size=kernel_size, strides=strides, padding=padding)(input_tensor)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation(activation=activation)(x)

        return x

    def build_encoder(self):

        input_shape = (self.length_TS, self.n_joints, self.dim)
        input_layer = tf.keras.layers.Input(input_shape)

        reshape_layer = tf.keras.layers.Reshape(target_shape=(
            self.length_TS, self.n_joints * self.dim))(input_layer)

        kernel_size = self.kernel_size  # kernel size = 40

        self.conv1 = self.conv_block(input_tensor=reshape_layer, n_filters=self.n_filters,
                                     kernel_size=kernel_size)

        kernel_size = self.kernel_size // 2  # kernel size = 20

        self.conv2 = self.conv_block(input_tensor=self.conv1, n_filters=self.n_filters,
                                     kernel_size=kernel_size)

        kernel_size = self.kernel_size // 4  # kernel size = 10

        self.conv3 = self.conv_block(input_tensor=self.conv2, n_filters=self.n_filters,
                                     kernel_size=kernel_size)

        self.shape_before_flatten = tf.keras.backend.int_shape(self.conv3)[1:]

        flatten_layer = tf.keras.layers.Flatten()(self.conv3)

        mu_layer = tf.keras.layers.Dense(
            units=self.bottleneck_size)(flatten_layer)
        var_layer = tf.keras.layers.Dense(
            units=self.bottleneck_size)(flatten_layer)

        self.encoder = tf.keras.models.Model(
            inputs=input_layer, outputs=[mu_layer, var_layer])

    def deconv_block(self, input_tensor, n_filters, kernel_size, activation='relu', strides=1, padding='same'):

        x = tf.keras.layers.Conv1DTranspose(
            filters=n_filters, kernel_size=kernel_size, strides=strides, padding=padding)(input_tensor)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation(activation=activation)(x)

        return x

    def build_decoder(self):

        input_shape = (self.bottleneck_size,)
        input_layer = tf.keras.layers.Input(input_shape)

        dense_layer = tf.keras.layers.Dense(
            units=np.prod(self.shape_before_flatten))(input_layer)
        reshape_layer = tf.keras.layers.Reshape(
            target_shape=self.shape_before_flatten)(dense_layer)

        kernel_size = self.kernel_size // 4  # kernel size = 10

        conv1T = self.deconv_block(input_tensor=reshape_layer, n_filters=self.n_filters,
                                   kernel_size=kernel_size)

        kernel_size = self.kernel_size // 2  # kernel size = 20

        conv2T = self.deconv_block(input_tensor=conv1T, n_filters=self.n_filters,
                                   kernel_size=kernel_size)

        kernel_size = self.kernel_size  # kernel size = 40

        conv3T = self.deconv_block(input_tensor=conv2T, n_filters=self.n_filters,
                                   kernel_size=kernel_size)

        reconstruction_layer = tf.keras.layers.Conv1DTranspose(
            filters=self.n_joints*self.dim, kernel_size=1, padding='same')(conv3T)

        output_layer = tf.keras.layers.Reshape(
            target_shape=(self.length_TS, self.n_joints, self.dim))(reconstruction_layer)

        self.decoder = tf.keras.models.Model(
            inputs=input_layer, outputs=output_layer)

    def reconstruction_loss(self, ytrue, ypred):

        ytrue = tf.keras.backend.cast(ytrue, dtype=ypred.dtype)
        return tf.keras.backend.mean(tf.keras.backend.sum(tf.keras.backend.square(ytrue-ypred), axis=1), axis=[1, 2])

    def kl_loss(self, mu, var):

        return -0.5 * tf.keras.backend.sum(1 + var - tf.keras.backend.square(mu) - tf.keras.backend.exp(var), axis=1)

    @tf.function
    def train_step(self, xtrain):

        with tf.GradientTape() as Encoder, tf.GradientTape() as Decoder:

            mu, var = self.encoder(xtrain, training=True) # Encode samples into Gaussian distribution
            latent_space = self.sampling(mu_var=[mu, var]) # Sample from this Gaussian distribution
            reconstructed_samples = self.decoder(latent_space, training=True) # Decode the sampled points to reconstruct the skeletton

            loss_rec = self.reconstruction_loss(
                ytrue=xtrain, ypred=reconstructed_samples)

            loss_kl = self.kl_loss(mu=mu, var=var)

            total_loss = tf.keras.backend.mean(
                self.w_rec * loss_rec + self.w_kl * loss_kl)

        gradients_encoder = Encoder.gradient(
            total_loss, self.encoder.trainable_variables)

        gradients_decoder = Decoder.gradient(
            total_loss, self.decoder.trainable_variables)

        self.optimizer.apply_gradients(
            zip(gradients_encoder, self.encoder.trainable_variables))

        self.optimizer.apply_gradients(
            zip(gradients_decoder, self.decoder.trainable_variables))

        return tf.keras.backend.mean(loss_rec), tf.keras.backend.mean(loss_kl), total_loss

    def generate_batch_indices(self, n):

        all_indices = np.random.permutation(n)

        if self.batch_size >= n:
            return [all_indices]

        remainder_batch_size = n % self.batch_size
        number_batches = n // self.batch_size

        batch_indices_list = np.array_split(
            ary=all_indices[:n-remainder_batch_size], indices_or_sections=number_batches)

        if remainder_batch_size > 0:
            batch_indices_list.append(all_indices[n-remainder_batch_size:])

        return batch_indices_list


    def train_model(self, xtrain, ytrain=None): # ytrain will not be used in this model

        batch_indices_list = self.generate_batch_indices(n=len(xtrain))
        number_batches = len(batch_indices_list)

        loss = []
        loss_kl = []
        loss_rec = []

        self.reduce_lr.on_train_begin()

        min_loss = 1e9

        for _epoch in range(self.n_epochs):

            loss_value = 0.0
            loss_kl_value = 0.0
            loss_rec_value = 0.0

            for batch_indices in batch_indices_list:

                loss_rec_tf, loss_kl_tf, loss_tf = self.train_step(
                    xtrain=xtrain[batch_indices])

                loss_value = loss_value + loss_tf.numpy()
                loss_kl_value = loss_kl_value + loss_kl_tf.numpy()
                loss_rec_value = loss_rec_value + loss_rec_tf.numpy()

            loss.append(loss_value / (number_batches * 1.0))
            loss_kl.append(loss_kl_value / (number_batches * 1.0))
            loss_rec.append(loss_rec_value / (number_batches * 1.0))

            if self.verbose_training:

                print("epoch: ", _epoch, ' total loss: ',
                    loss[-1], ' rec loss: ', loss_rec[-1],
                    ' kl loss: ', loss_kl[-1])

            self.reduce_lr.on_epoch_end(epoch=_epoch, loss=loss[-1])

            if loss[-1] < min_loss:

                min_loss = loss[-1]

                self.encoder.save(self.output_directory + 'best_encoder.hdf5')
                self.decoder.save(self.output_directory + 'best_decoder.hdf5')

        self.encoder.save(self.output_directory + 'last_encoder.hdf5')
        self.decoder.save(self.output_directory + 'last_decoder.hdf5')

        plt.figure(figsize=(20, 10))

        plt.plot(loss_rec, color='green', lw=3, label='reconstruction loss')
        plt.plot(loss_kl, color='blue', lw=3, label='kl loss')
        plt.plot(loss, color='black', lw=3, label='total loss')

        plt.legend()

        plt.savefig(self.output_directory + 'loss.pdf')
        plt.cla()
        plt.clf()

        tf.keras.backend.clear_session()

    def generate_array_of_colors(self, n):

        colors = ["#"+''.join([random.choice('0123456789ABCDEF') for _ in range(6)])
                  for __ in range(n)]
        return colors

    def latent_space_visualization(self, xtrain, ytrain):

        encoder = tf.keras.models.load_model(
            self.output_directory + 'best_encoder.hdf5', compile=False)

        self.mu, self.var = encoder.predict(xtrain)

        self.latent_space = self.sampling(mu_var=[self.mu, self.var])
        self.latent_space = np.asarray(self.latent_space)

        colors = self.generate_array_of_colors(n=self.n_classes)

        plt.figure(figsize=(20,10))

        if self.bottleneck_size == 2:

            for c in range(self.n_classes):

                c_ind = np.where(ytrain == c)[0]
                plt.scatter(self.mu[c_ind, 0], self.mu[c_ind,1],
                            c=colors[c], s=200, label='class '+str(c))

            plt.savefig(self.output_directory + 'mean.pdf')
            plt.cla()

            for c in range(self.n_classes):

                c_ind = np.where(ytrain == c)[0]
                plt.scatter(self.var[c_ind, 0], self.var[c_ind,1],
                            c=colors[c], s=200, label='class '+str(c))

            plt.savefig(self.output_directory + 'var.pdf')
            plt.cla()

            for c in range(self.n_classes):
    
                c_ind = np.where(ytrain == c)[0]
                plt.scatter(self.latent_space[c_ind, 0], self.latent_space[c_ind,
                            1], c=colors[c], s=200, label='class '+str(c))
            plt.legend()
            plt.savefig(self.output_directory + 'TwoD_latent_space.pdf')
            plt.cla()

        else:

            pca = PCA(n_components=2)
            mu_2d = pca.fit_transform(self.mu)

            for c in range(self.n_classes):

                c_ind = np.where(ytrain == c)[0]
                plt.scatter(mu_2d[c_ind, 0], mu_2d[c_ind,1],
                            c=colors[c], s=200, label='class '+str(c))

            plt.savefig(self.output_directory + 'PCA_mean.pdf')
            plt.cla()

            tsne = TSNE(n_components=2)
            mu_2d = tsne.fit_transform(self.mu)

            for c in range(self.n_classes):

                c_ind = np.where(ytrain == c)[0]
                plt.scatter(mu_2d[c_ind, 0], mu_2d[c_ind,1],
                            c=colors[c], s=200, label='class '+str(c))

            plt.savefig(self.output_directory + 'TSNE_mean.pdf')
            plt.cla()

            pca = PCA(n_components=2)
            var_2d = pca.fit_transform(self.var)

            for c in range(self.n_classes):

                c_ind = np.where(ytrain == c)[0]
                plt.scatter(var_2d[c_ind, 0], var_2d[c_ind,1],
                            c=colors[c], s=200, label='class '+str(c))

            plt.savefig(self.output_directory + 'PCA_var.pdf')
            plt.cla()

            tsne = TSNE(n_components=2)
            var_2d = tsne.fit_transform(self.var)

            for c in range(self.n_classes):

                c_ind = np.where(ytrain == c)[0]
                plt.scatter(var_2d[c_ind, 0], var_2d[c_ind,1],
                            c=colors[c], s=200, label='class '+str(c))

            plt.savefig(self.output_directory + 'TSNE_var.pdf')
            plt.cla()

            pca = PCA(n_components=2)
            latent_space_2d = pca.fit_transform(self.latent_space)

            for c in range(self.n_classes):

                c_ind = np.where(ytrain == c)[0]
                plt.scatter(latent_space_2d[c_ind, 0], latent_space_2d[c_ind,1],
                            c=colors[c], s=200, label='class '+str(c))
            plt.legend()
            plt.savefig(self.output_directory + 'PCA_latent_space.pdf')
            plt.cla()

            tsne = TSNE(n_components=2)
            latent_space_2d = tsne.fit_transform(self.latent_space)

            for c in range(self.n_classes):
    
                c_ind = np.where(ytrain == c)[0]
                plt.scatter(latent_space_2d[c_ind, 0], latent_space_2d[c_ind,1],
                            c=colors[c], s=200, label='class '+str(c))
            plt.legend()
            plt.savefig(self.output_directory + 'TSNE_latent_space.pdf')
            plt.cla()

        plt.clf()

    def generate_samples_class(self, xtrain=None, ytrain=None, n=1, c=0):

        assert(ytrain is not None)

        try:
            latent_space_c = self.latent_space[ytrain == c]
        
        except:

            assert(xtrain is not None)
            encoder = tf.keras.models.load_model(
            self.output_directory + 'best_encoder.hdf5', compile=False)

            mu, var = encoder.predict(xtrain[ytrain == c])

            latent_space_c = self.sampling(mu_var=[mu, var])

        gm = GM(n_components=1).fit(latent_space_c)

        means = np.asarray(gm.means_).reshape((self.bottleneck_size,))
        covariances = np.asarray(gm.covariances_).reshape((self.bottleneck_size, self.bottleneck_size))

        new_latent_space_c = np.random.multivariate_normal(mean=means,cov=covariances, size=n)
        
        decoder = tf.keras.models.load_model(
            self.output_directory + 'best_decoder.hdf5', compile=False)

        generated_samples_c = decoder.predict(new_latent_space_c)
        generated_samples_c = np.asarray(generated_samples_c).reshape(
            (n, self.length_TS, self.n_joints, self.dim))

        return generated_samples_c

    def generate_samples(self, y_to_generate_from, xtrain=None, ytrain=None, factor=1, with_labels=True):

        assert(ytrain is not None)

        n = len(y_to_generate_from)

        generated_samples = np.zeros(shape=(n * factor, self.length_TS, self.n_joints, self.dim))

        if with_labels:
            labels_generated_samples = np.zeros(shape=(n*factor,))

        n_generated = 0

        for c in range(self.n_classes):

            n_to_generate = factor * len(y_to_generate_from[y_to_generate_from == c])
            
            generated_samples_c = self.generate_samples_class(xtrain=xtrain,
                                                              ytrain=ytrain,
                                                              n=n_to_generate,
                                                              c=c)
            
            generated_samples[n_generated:n_generated+n_to_generate] = generated_samples_c

            if with_labels:
                labels_generated_samples[n_generated:n_generated+n_to_generate] = c

            n_generated = n_generated + n_to_generate

        if with_labels:
            return generated_samples, labels_generated_samples

        return generated_samples
