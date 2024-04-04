import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

import numpy as np
import tensorflow as tf
gpus = tf.config.list_physical_devices(device_type = 'GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)
import pandas as pd
import matplotlib.pyplot as plt
import argparse
from distutils.util import strtobool

from utils.utils import *
from utils.show_skeletton import plot_skeleton
from utils.classes_map import classes_map

from generators.VAE import VAE
from generators.SVAE import SVAE

def get_args():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--dataset',
        help="Which dataset to use.",
        type=str,
        default='HumanAct12'
    )

    parser.add_argument(
        '--generative-model',
        help="Which generative model to use.",
        type=str,
        choices=['VAE','SVAE'],
        default='VAE'
    )

    parser.add_argument(
        '--run',
        type=int,
        default=0
    )

    parser.add_argument(
        '--plot-skeletons',
        type=lambda x: bool(strtobool(x)),
        default=False
    )

    parser.add_argument(
        '--weight-rec',
        type=float,
        default=1.0
    )

    parser.add_argument(
        '--weight-kl',
        type=float,
        default=1.0
    )

    parser.add_argument(
        '--weight-cls',
        type=float,
        default=1.0
    )

    parser.add_argument(
        '--save-skeletons',
        type=lambda x: bool(strtobool(x)),
        default=True
    )

    parser.add_argument(
        '--class-generate',
        type=int,
        default=0
    )

    parser.add_argument(
        '--n-samples',
        type=int,
        default=1
    )

    parser.add_argument(
        '--best-predictions',
        type=int,
        default=3
    )

    parser.add_argument(
        '--output-directory',
        type=str,
        default='results/'
    )

    args = parser.parse_args()

    return args

if __name__ == "__main__":

    args = get_args()

    X, Y, S = load_data(root_dir='datasets/' + args.dataset + '/')
    
    n_X, _,_, _,_, _,_ = normalize_skeletons(X=X)

    length_TS = int(X.shape[1])
    n_joints = int(X.shape[2])
    dim = int(X.shape[3])

    n_classes = len(np.unique(Y))

    GRU_path = args.output_directory + 'GRU/on_real/all/run_0/best_model.hdf5'
    GRU_model = tf.keras.models.load_model(GRU_path, compile=False)

    output_directory_results = args.output_directory
    output_directory_gen_models = output_directory_results + 'Generative_models/'

    output_directory_gen_model = output_directory_gen_models + args.generative_model + '/'

    if args.generative_model != 'SVAE':
        output_directory_gen_weight = output_directory_gen_model + 'Wrec_' + str(args.weight_rec) + '_Wkl_' + str(args.weight_kl) + '/'

    else:
        output_directory_gen_weight = output_directory_gen_model + 'Wrec_' + str(args.weight_rec) + '_Wkl_' + str(args.weight_kl) + '_Wcls_' + str(args.weight_cls) + '/'

    weights_loss = {
        'Wrec' : args.weight_rec,
        'Wkl' : args.weight_kl,
        'Wcls' : args.weight_cls
    }

    output_directory_split = output_directory_gen_weight + 'all/'

    output_directory_run = output_directory_split + 'run_' + str(args.run) + '/'

    output_directory_skeletons = output_directory_split + 'skeletons/'
    create_directory(output_directory_skeletons)

    output_directory_skeletons_class = output_directory_skeletons + 'class_' + str(args.class_generate) + '/'
    create_directory(output_directory_skeletons_class)

    if args.generative_model == 'VAE':
        generator = VAE(output_directory=output_directory_run,
                            length_TS=length_TS,
                            n_joints=n_joints,
                            dim=dim,
                            n_classes=n_classes,
                            w_rec=weights_loss['Wrec'],
                            w_kl=weights_loss['Wkl'])
    
    elif args.generative_model == 'SVAE':
        generator = SVAE(output_directory=output_directory_run,
                        length_TS=length_TS,
                        n_joints=n_joints,
                        dim=dim,
                        n_classes=n_classes,
                        w_rec=weights_loss['Wrec'],
                        w_kl=weights_loss['Wkl'],
                        w_cls=weights_loss['Wcls'])

    print("Generating with ",args.generative_model," with weights ",weights_loss)

    generated_samples = generator.generate_samples_class(
        xtrain=n_X,
        ytrain=Y,
        n=args.n_samples,
        c=args.class_generate
    )

    if args.save_skeletons:
        np.save(file=output_directory_skeletons_class + 'skeletons.npy', arr=generated_samples)

    if args.plot_skeletons:

        ypred = GRU_model.predict(generated_samples)

        for i in range(args.n_samples):

            print("Plotting skeleton #", i)

            output_directory_skeletons_class_n = output_directory_skeletons_class + 'generation_' + str(i) + '/'
            create_directory(output_directory_skeletons_class_n)

            predictions_sorted_indices = np.argsort(ypred[i])[::-1]
            
            if predictions_sorted_indices[0] == args.class_generate:
                title = 'Correctly classified. Top 3 predictions :' + '\n'
            else:
                title = 'Wrongly classified. Top 3 predictions :' + '\n'
            
            n_best_predictions = args.best_predictions

            if str(ypred[i, predictions_sorted_indices[0]]) == '1.0':
                n_best_predictions = 1

            for j in range(args.best_predictions):

                title = title + classes_map[args.dataset][predictions_sorted_indices[j]] + ' : ' + str(ypred[i,predictions_sorted_indices[j]])
                if j < 2:
                    title = title + '\n'

            plot_skeleton(x=generated_samples[i],
                          output_directory=output_directory_skeletons_class_n,
                          title=title)