import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import scipy

from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import euclidean_distances

class FID_DIVERSITY_CALCULATOR:

    def __init__(self,runs_diversity=5, Sd=200):

        self.runs_diversity = runs_diversity
        self.Sd = Sd

    def remove_classification_layer(self, model):

        input_model = model.input
        output_model = model.layers[-2].output

        return tf.keras.models.Model(inputs=input_model, outputs=output_model)

    def get_FID(self, classifier_dir, x, y):

        classifier = tf.keras.models.load_model(classifier_dir)
        classifier = self.remove_classification_layer(model=classifier)

        x_batches = self.split_to_batches(x=x)
        y_batches = self.split_to_batches(x=y)

        
        x_latent = []
        y_latent = []

        for batch in x_batches:
            x_latent.append(np.asarray(classifier(batch)))
        
        for batch in y_batches:
            y_latent.append(np.asarray(classifier(batch)))
        
        x_latent = self.rejoin_batches(x=x_latent, n=int(x.shape[0]))
        y_latent = self.rejoin_batches(x=y_latent, n=int(y.shape[0]))

        mean_x = np.mean(x_latent, axis=0)
        cov_x = np.cov(x_latent, rowvar=False)

        mean_y = np.mean(y_latent, axis=0)
        cov_y = np.cov(y_latent, rowvar=False)

        diff_means = np.sum((mean_x - mean_y) ** 2.0)
        cov_prod = scipy.linalg.sqrtm(cov_x.dot(cov_y))

        if np.iscomplexobj(cov_prod):
            cov_prod = cov_prod.real
        
        fid = diff_means + np.trace(cov_x + cov_y - 2.0 * cov_prod)

        return fid
    
    def get_DIVERSITY(self, classifier_dir, x):

        classifier = tf.keras.models.load_model(classifier_dir)
        classifier = self.remove_classification_layer(model=classifier)
        
        x_batches = self.split_to_batches(x=x)

        x_latent = []

        for batch in x_batches:
            x_latent.append(np.asarray(classifier(batch)))
        
        x_latent = self.rejoin_batches(x=x_latent, n=int(x.shape[0]))

        if self.Sd > len(x):
            Sd = len(x)
        else:
            Sd = self.Sd

        divs = []

        for _ in range(self.runs_diversity):

            all_indices = np.arange(len(x))
            
            V = x_latent[np.random.choice(a=all_indices, size=Sd)]
            V_prime = x_latent[np.random.choice(a=all_indices, size=Sd)]

            div = np.mean(np.linalg.norm(V-V_prime, axis=1))
            divs.append(div)
        
        return np.mean(divs)

    def split_to_batches(self, x, batch_size=128):

        n = int(x.shape[0])
        batches = []
        for i in range(0,n-batch_size+1,batch_size):
            batches.append(x[i:i+batch_size])
        if n % batch_size > 0:
            batches.append(x[i+batch_size:n])
        return batches

    def rejoin_batches(self, x, n):

        m = len(x)
        d = x[0].shape[-1]
        x_rejoin = np.zeros(shape=(n,d))
        filled = 0

        for i in range(m):
            _stop = len(x[i])
            x_rejoin[filled:filled+_stop,:] = x[i]
            filled += len(x[i])

        return x_rejoin