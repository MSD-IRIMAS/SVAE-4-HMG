import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

from sklearn.preprocessing import OneHotEncoder as OHE
from sklearn.metrics import accuracy_score

a = tf.random.normal((1,))
print("Done initialization.")

# gpus = tf.config.list_physical_devices('GPU')
# for gpu in gpus:
#     tf.config.experimental.set_memory_growth(gpu, True)

class GRU_CLASSIFIER:

    def __init__(self, output_directory, length_TS, n_joints, dim, n_classes,
                       batch_size=32, n_epochs=2000, activation='tanh', verbose=True,
                       hidden_units_gru=128, hidden_units_fc=30):
    
        self.output_directory = output_directory
        
        self.length_TS = length_TS
        self.n_joints = n_joints
        self.dim = dim

        self.n_classes = n_classes

        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.verbose = verbose

        self.activation = activation
        self.hidden_units_gru = hidden_units_gru
        self.hidden_units_fc = hidden_units_fc

        self.build_model()
    
    def build_model(self):

        input_shape = (self.length_TS, self.n_joints, self.dim)

        input_layer = tf.keras.layers.Input(input_shape)
        reshape_layer = tf.keras.layers.Reshape(target_shape=(self.length_TS, self.n_joints * self.dim))(input_layer)

        gru_1 = tf.keras.layers.GRU(units=self.hidden_units_gru, activation=self.activation,
                                       return_sequences=True)(reshape_layer)
        
        gru_2 = tf.keras.layers.GRU(units=self.hidden_units_gru, activation=self.activation,
                                       return_sequences=True)(gru_1)

        hidden_layer = tf.keras.layers.Dense(units=self.hidden_units_fc, activation=self.activation)(gru_2[:,-1,:])

        output_layer = tf.keras.layers.Dense(units=self.n_classes, activation='softmax')(hidden_layer)

        self.model = tf.keras.models.Model(inputs=input_layer, outputs=output_layer)

        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=50,
                                                         min_lr=1e-4)
        model_checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=self.output_directory + 'best_model.hdf5',
                                                              monitor='loss', save_best_only=True)
        self.callbacks = [reduce_lr, model_checkpoint]

        self.model.compile(loss='categorical_crossentropy',optimizer='Adam',metrics=['accuracy'])

    def fit(self, xtrain, ytrain, xval=None, yval=None, plot_test=False):

        ohe = OHE(sparse=False)
        ytrain = np.expand_dims(ytrain, axis=1)
        ytrain = ohe.fit_transform(ytrain)
        
        if plot_test:

            ohe = OHE(sparse=False)
            yval = np.expand_dims(yval, axis=1)
            yval = ohe.fit_transform(yval)
        
            hist = self.model.fit(xtrain, ytrain, batch_size=self.batch_size, epochs=self.n_epochs,
                                  callbacks=self.callbacks, validation_data=(xval, yval))
        
        else:

            hist = self.model.fit(xtrain, ytrain, batch_size=self.batch_size, epochs=self.n_epochs,
                                  callbacks=self.callbacks)

        plt.figure(figsize=(20,10))

        plt.plot(hist.history['loss'], color='blue', lw=3, label='Training Loss')
        if plot_test:
            plt.plot(hist.history['val_loss'], color='red', lw=3, label='Validation Loss')
            
        plt.legend()
        plt.savefig(self.output_directory + 'loss.pdf')

        plt.cla()
        plt.plot(hist.history['accuracy'], color='blue', lw=3, label='Training Accuracy')
        if plot_test:
            plt.plot(hist.history['val_accuracy'], color='red', lw=3, label='Validation Accuracy')
        
        plt.legend()
        plt.savefig(self.output_directory + 'accuracy.pdf')

        plt.cla()
        plt.clf()
    
    def predict(self, xtest, ytest):
    
        model = tf.keras.models.load_model(self.output_directory + 'best_model.hdf5')

        ypred = model.predict(xtest)
        ypred = np.argmax(ypred, axis=1)

        return accuracy_score(y_true=ytest, y_pred=ypred)