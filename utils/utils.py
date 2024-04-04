import tensorflow as tf
import numpy as np
import os

def load_data(root_dir):

    X = np.load(root_dir+'X.npy')
    Y = np.load(root_dir+'Y.npy')

    try:
        S = np.load(root_dir+'S.npy')

    except FileNotFoundError:
        S = np.arange(len(Y))

    return X, Y, S

def normalize_skeletons(X, min_X=None, max_X=None, min_Y=None, max_Y=None, min_Z=None, max_Z=None):

    n_X = np.zeros(shape=X.shape)

    if min_X is None:
        min_X = np.min(X[:,:,:,0])
    
    if max_X is None:
        max_X = np.max(X[:,:,:,0])

    if min_Y is None:
        min_Y = np.min(X[:,:,:,1])
    
    if max_Y is None:
        max_Y = np.max(X[:,:,:,1])

    if min_Z is None:
        min_Z = np.min(X[:,:,:,2])
    
    if max_Z is None:
        max_Z = np.max(X[:,:,:,2])

    n_X[:,:,:,0] = (X[:,:,:,0] - min_X) / (1.0 * (max_X - min_X))
    n_X[:,:,:,1] = (X[:,:,:,1] - min_Y) / (1.0 * (max_Y - min_Y))
    n_X[:,:,:,2] = (X[:,:,:,2] - min_Z) / (1.0 * (max_Z - min_Z))

    return n_X, min_X, max_X, min_Y, max_Y, min_Z, max_Z

def create_directory(directory_path):

    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

def get_inverse(S, leave_out_split):

    all_S = list(np.unique(S))

    for s in leave_out_split:
        all_S.remove(s)
    
    return all_S

def get_n_runs(dir):

    runs_dirs = []
    for _, dirs, _ in os.walk(dir):
        runs_dirs = dirs.copy()
        break
    
    runs = []
    for runs_dir in runs_dirs:
        if runs_dir[0:3] == 'run':
            runs.append(runs_dir)
    
    return len(runs)

def get_dirs(dir):

    for _, dirs, _ in os.walk(dir):
        return dirs

def get_weights_loss(dir):

    weights = dir.split('_')
    weights_dict = {}

    for i in range(len(weights)):

        if weights[i][0] == 'W':
            weights_dict[weights[i]] = float(weights[i+1])
            
    return weights_dict