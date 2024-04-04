import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import random

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.mixture import GaussianMixture as GM
from sklearn.preprocessing import OneHotEncoder as OHE

from utils.custom_callback import CustomReduceLRoP

#gpus = tf.config.list_physical_devices('GPU')
#for gpu in gpus:
#    tf.config.experimental.set_memory_growth(gpu, True)

class SVAE:

    def __init__(self, output_directory, length_TS, n_joints,
                       dim, n_classes, batch_size=32, n_epochs=2000,
                       n_filters=128, kernel_size=40, activation='relu',
                       bottleneck_size=16, use_weighted_loss=True,
                       w_rec=0.29, w_kl=0.001, w_cls=0.7,
                       show_summaries=False, verbose_training=True):
        
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
        self.w_cls = w_cls

        if not self.use_weighted_loss:

            self.w_rec, self.kl, self.w_cls = 1.0, 1.0, 1.0

        self.show_summaries = show_summaries
        self.verbose_training = verbose_training

        self.build_encoder()
        if self.show_summaries:
            self.encoder.summary()

        self.build_decoder()
        if self.show_summaries:
            self.decoder.summary()

        self.build_classifier()
        if self.show_summaries:
            self.classifier.summary()

        self.optimizer = tf.keras.optimizers.legacy.Adam()

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

    def build_classifier(self):

        input_shape = (self.bottleneck_size,)
        input_layer = tf.keras.layers.Input(input_shape)

        x = tf.keras.layers.Dense(units=self.bottleneck_size // 2)(input_layer)
        x = tf.keras.layers.Dropout(0.5)(x)
        x = tf.keras.layers.Dense(units=self.n_classes)(x)

        output_layer = tf.keras.layers.Dense(units=self.n_classes, activation='softmax')(x)

        self.classifier = tf.keras.models.Model(inputs=input_layer, outputs=output_layer)

    def reconstruction_loss(self, ytrue, ypred):

        ytrue = tf.keras.backend.cast(ytrue, dtype=ypred.dtype)
        return tf.keras.backend.mean(tf.keras.backend.sum(tf.keras.backend.square(ytrue-ypred), axis=1), axis=[1, 2])

    def kl_loss(self, mu, var):

        return -0.5 * tf.keras.backend.sum(1 + var - tf.keras.backend.square(mu) - tf.keras.backend.exp(var), axis=1)

    def cls_loss(self, ytrue, ypred):

        return tf.keras.losses.categorical_crossentropy(ytrue, ypred)

    @tf.function
    def train_step(self, xtrain, ytrain):

        with tf.GradientTape() as Encoder, tf.GradientTape() as Decoder, tf.GradientTape() as Classifier:

            mu, var = self.encoder(xtrain, training=True) # Encode samples into Gaussian distribution
            
            latent_space = self.sampling(mu_var=[mu, var]) # Sample from this Gaussian distribution
            ypred = self.classifier(latent_space, training=True) # predict the class using the classifier

            reconstructed_samples = self.decoder(latent_space, training=True) # Decode the sampled points to reconstruct the skeletton

            loss_rec = self.reconstruction_loss(
                ytrue=xtrain, ypred=reconstructed_samples)

            loss_kl = self.kl_loss(mu=mu, var=var)

            loss_cls = self.cls_loss(ytrue=ytrain, ypred=ypred)

            total_loss = tf.keras.backend.mean(
                self.w_rec * loss_rec + self.w_kl * loss_kl + self.w_cls * loss_cls)

        gradients_encoder = Encoder.gradient(
            total_loss, self.encoder.trainable_variables)

        gradients_decoder = Decoder.gradient(
            total_loss, self.decoder.trainable_variables)

        gradients_classifier = Classifier.gradient(
            total_loss, self.classifier.trainable_variables)

        self.optimizer.apply_gradients(
            zip(gradients_encoder, self.encoder.trainable_variables))

        self.optimizer.apply_gradients(
            zip(gradients_decoder, self.decoder.trainable_variables))
        
        self.optimizer.apply_gradients(
            zip(gradients_classifier, self.classifier.trainable_variables))

        return tf.keras.backend.mean(loss_rec), tf.keras.backend.mean(loss_kl), tf.keras.backend.mean(loss_cls), total_loss

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

    def train_model(self, xtrain, ytrain):

        ohe = OHE(sparse=False)

        ytrain = np.expand_dims(ytrain, axis=1)
        ytrain = ohe.fit_transform(ytrain)

        batch_indices_list = self.generate_batch_indices(n=len(xtrain))
        number_batches = len(batch_indices_list)

        loss = []
        loss_kl = []
        loss_rec = []
        loss_cls = []

        self.reduce_lr.on_train_begin()

        min_loss = 1e9
        found_one_best = False

        for _epoch in range(self.n_epochs):

            loss_value = 0.0
            loss_kl_value = 0.0
            loss_rec_value = 0.0
            loss_cls_value = 0.0

            for batch_indices in batch_indices_list:

                loss_rec_tf, loss_kl_tf, loss_cls_tf, loss_tf = self.train_step(
                    xtrain=xtrain[batch_indices], ytrain=ytrain[batch_indices])

                loss_value = loss_value + loss_tf.numpy()
                loss_kl_value = loss_kl_value + loss_kl_tf.numpy()
                loss_rec_value = loss_rec_value + loss_rec_tf.numpy()
                loss_cls_value = loss_cls_value + loss_cls_tf.numpy()

            loss.append(loss_value / (number_batches * 1.0))
            loss_kl.append(loss_kl_value / (number_batches * 1.0))
            loss_rec.append(loss_rec_value / (number_batches * 1.0))
            loss_cls.append(loss_cls_value / (number_batches * 1.0))

            if self.verbose_training:

                print("epoch: ", _epoch, ' total loss: ',
                    loss[-1], ' rec loss: ', loss_rec[-1],
                    ' kl loss: ', loss_kl[-1], ' cls loss: ',
                    loss_cls[-1])

            self.reduce_lr.on_epoch_end(epoch=_epoch, loss=loss[-1])

            if loss[-1] < min_loss:

                min_loss = loss[-1]
                found_one_best = True

                self.encoder.save(self.output_directory + 'best_encoder.hdf5')
                self.decoder.save(self.output_directory + 'best_decoder.hdf5')
                self.classifier.save(self.output_directory + 'best_classifier.hdf5')

        self.encoder.save(self.output_directory + 'last_encoder.hdf5')
        self.decoder.save(self.output_directory + 'last_decoder.hdf5')
        self.classifier.save(self.output_directory + 'last_classifier.hdf5')
        
        if not found_one_best:
            
            self.encoder.save(self.output_directory + 'best_encoder.hdf5')
            self.decoder.save(self.output_directory + 'best_decoder.hdf5')
            self.classifier.save(self.output_directory + 'best_classifier.hdf5')
            
        plt.figure(figsize=(20, 10))

        plt.plot(loss_rec, color='green', lw=3, label='reconstruction loss')
        plt.plot(loss_kl, color='blue', lw=3, label='kl loss')
        plt.plot(loss_cls, color='red', lw=3, label='classificatio loss')
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

        generated_samples = np.zeros(shape=(int(n * factor), self.length_TS, self.n_joints, self.dim))
        
        if with_labels:
            labels_generated_samples = np.zeros(shape=(n*factor,))

        n_generated = 0

        for c in range(self.n_classes):

            n_to_generate = int(factor * len(y_to_generate_from[y_to_generate_from == c]))
            
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
    
    def get_most_populated_class(self, y):

        labels_unique, labels_count = np.unique(y, return_counts=True)

        return labels_unique[np.argmax(labels_count)]

    def get_how_much_to_generate(self, y, most_populated_class):

        n_to_generate_per_class = np.zeros(shape=(len(np.unique(y)),))

        for c in range(len(np.unique(y))):
            
            if c != most_populated_class:
                n_to_generate_per_class[c] = len(y[y == most_populated_class]) - len(y[y == c])
        
        return n_to_generate_per_class

    def generate_samples_to_fix_labeling(self, xtrain=None, ytrain=None):

        assert(xtrain is not None)
        assert(ytrain is not None)

        most_populated_class = self.get_most_populated_class(y=ytrain)
        n_to_generate_per_class = self.get_how_much_to_generate(y=ytrain, most_populated_class=most_populated_class)

        generated_samples = np.zeros(shape=(int(np.sum(n_to_generate_per_class)), self.length_TS, self.n_joints, self.dim))
        labels_generated = np.zeros(shape=(int(np.sum(n_to_generate_per_class)),))

        n_generated = 0

        for c in range(len(np.unique(ytrain))):

            if n_to_generate_per_class[c] > 0:

                generated_samples_c = self.generate_samples_class(xtrain=xtrain, ytrain=ytrain, c=c, n=int(n_to_generate_per_class[c]))
                generated_samples[n_generated:n_generated+int(n_to_generate_per_class[c])] = generated_samples_c

                labels_generated[n_generated:n_generated+int(n_to_generate_per_class[c])] = c

                n_generated = n_generated + int(n_to_generate_per_class[c])
        
        return generated_samples, labels_generated