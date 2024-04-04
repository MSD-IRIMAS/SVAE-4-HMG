import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

import pandas as pd
import numpy as np
import argparse

from sklearn.preprocessing import LabelEncoder as LE
from sklearn.model_selection import train_test_split

from utils.utils import load_data, normalize_skeletons, create_directory
from generators.VAE import VAE
from generators.SVAE import SVAE

def get_args():

    parser = argparse.ArgumentParser(
    description="Choose which samples to train the GRU classifier on with the type of split.")

    parser.add_argument(
        '--generative-model',
        help="Which generative model to use if training on generated or augmented.",
        type=str,
        choices=['VAE','SVAE'],
        default='VAE',
    )

    parser.add_argument(
        '--dataset',
        help="Which dataset to use.",
        type=str,
        default='HumanAct12'
    )

    parser.add_argument(
        '--output-directory',
        type=str,
        default='results/'
    )

    parser.add_argument(
        '--runs',
        help="Number of experiments to do.",
        type=int,
        default=5
    )

    parser.add_argument(
        '--split',
        help="Type of split to do.",
        type=str,
        choices=['all','train_test'],
        default='all'
    )

    parser.add_argument(
        '--subject',
        help="Which subject to train on if the split is chosen to be cross subject. If you want \
            to train on all the subjects in a cross subject method use -1 as subject ID, else please \
                specify a valid subject ID.",
        type=int,
        default=-1
    )

    parser.add_argument(
        '--weight-rec',
        help="Weight for the reconstruction loss.",
        type=float,
        default=1.0
    )

    parser.add_argument(
        '--weight-kl',
        help="Weight for the kl loss.",
        type=float,
        default=1.0
    )

    parser.add_argument(
        '--weight-cls',
        help="Weight for the classification loss.",
        type=float,
        default=1.0
    )

    parser.add_argument(
        '--n-epochs',
        help="Number of epochs to train the generative model.",
        type=int,
        default=2000
    )

    args = parser.parse_args()

    return args
    
if __name__ == "__main__":

    args = get_args()

    output_directory_results = args.output_directory
    create_directory(output_directory_results)

    output_directory_gen_models = output_directory_results + 'Generative_models/'
    create_directory(output_directory_gen_models)

    output_directory_dataset = output_directory_gen_models + args.dataset + '/'
    create_directory(output_directory_dataset)

    output_directory_generator = output_directory_dataset + args.generative_model + '/'
    create_directory(output_directory_generator)

    if args.generative_model == 'SVAE':
        output_directory_weights_losses = output_directory_generator + 'Wrec_' + str(args.weight_rec) + '_Wkl_' + str(args.weight_kl) + '_Wcls_' + str(args.weight_cls) + '/'
    else:
        output_directory_weights_losses = output_directory_generator + 'Wrec_' + str(args.weight_rec) + '_Wkl_' + str(args.weight_kl) + '/'

    create_directory(output_directory_weights_losses)

    output_directory_split = output_directory_weights_losses + args.split + '/'
    create_directory(output_directory_split)

    dataset_dir = 'datasets/' + args.dataset + '/'
    X, Y, S = load_data(root_dir=dataset_dir)

    le = LE()
    Y = le.fit_transform(Y)

    le = LE()
    S = le.fit_transform(S)

    if args.split == 'cross_subject':
        if args.subject != -1:
            if args.subject < S.min() or args.subject > S.max():
                raise ValueError("Wrong subject, does not exist.")
    
    length_TS = int(X.shape[1])
    n_joints = int(X.shape[2])
    dim = int(X.shape[3])

    if args.split == 'all':

        n_X, _,_, _,_, _,_ = normalize_skeletons(X=X)
        n_classes = len(np.unique(Y))

        for _run in range(args.runs):

            output_directory_run = output_directory_split + 'run_' + str(_run) + '/'
            create_directory(output_directory_run)

            if args.generative_model == 'VAE':
                generator = VAE(output_directory=output_directory_run,
                                length_TS=length_TS,
                                n_joints=n_joints,
                                dim=dim,
                                n_classes=n_classes,
                                w_rec=args.weight_rec,
                                w_kl=args.weight_kl,
                                n_epochs=args.n_epochs)

            elif args.generative_model == 'SVAE':
                generator = SVAE(output_directory=output_directory_run,
                                length_TS=length_TS,
                                n_joints=n_joints,
                                dim=dim,
                                n_classes=n_classes,
                                w_rec=args.weight_rec,
                                w_kl=args.weight_kl,
                                w_cls=args.weight_cls,
                                n_epochs=args.n_epochs)
            
            generator.train_model(xtrain=n_X, ytrain=Y)             

    elif args.split == 'train_test':

        n_X, _,_, _,_, _,_ = normalize_skeletons(X=X)

        leave_out_splits = np.load('datasets/' + args.dataset + '/leave_out.npy')

        for _split, leave_out_split in enumerate(leave_out_splits):

            output_directory_n_split = output_directory_split + 'split_' + str(_split) + '/'
            create_directory(output_directory_n_split)

            X_split = n_X.copy()
            Y_split = Y.copy()
            S_split = S.copy()

            for s in leave_out_split:
                
                X_split = X_split[S_split != s]
                Y_split = Y_split[S_split != s]
                S_split = S_split[S_split != s]
            
            n_classes = len(np.unique(Y_split))

            for _run in range(args.runs):

                output_directory_run = output_directory_n_split + 'run_' + str(_run) + '/'
                create_directory(output_directory_run)

                if args.generative_model == 'VAE':
                    generator = VAE(output_directory=output_directory_run,
                                    length_TS=length_TS,
                                    n_joints=n_joints,
                                    dim=dim,
                                    n_classes=n_classes,
                                    w_rec=args.weight_rec,
                                    w_kl=args.weight_kl,
                                    n_epochs=args.n_epochs)

                elif args.generative_model == 'SVAE':
                    generator = SVAE(output_directory=output_directory_run,
                                    length_TS=length_TS,
                                    n_joints=n_joints,
                                    dim=dim,
                                    n_classes=n_classes,
                                    w_rec=args.weight_rec,
                                    w_kl=args.weight_kl,
                                    w_cls=args.weight_cls,
                                    n_epochs=args.n_epochs)
                
                if not os.path.exists(output_directory_run + 'loss.pdf'):
                    
                    generator.train_model(xtrain=X_split, ytrain=Y_split)
                    generator.latent_space_visualization(xtrain=X_split, ytrain=Y_split)