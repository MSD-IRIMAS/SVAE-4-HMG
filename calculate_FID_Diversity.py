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
import scipy
import argparse

from sklearn.model_selection import train_test_split

from utils.fid_div import FID_DIVERSITY_CALCULATOR
from utils.utils import load_data, create_directory, get_n_runs, get_dirs, get_weights_loss, normalize_skeletons

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
        '--output-directory',
        type=str,
        default='results/'
    )

    parser.add_argument(
        '--generative-model',
        help="Which generative model to use.",
        type=str,
        choices=['VAE','SVAE'],
        default='VAE'
    )

    parser.add_argument(
        '--on',
        help="On which data apply the metrics on.",
        type=str,
        choices=['real','generated'],
        default='generated'
    )

    parser.add_argument(
        '--n-generations',
        help="Number of generations to do per generative model pre-trained.",
        type=int,
        default=5
    )

    parser.add_argument(
        '--n-factors',
        help="Index of max facotrs, max_factor=2^n_factors",
        type=int,
        default=5
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

    output_directory_results = args.output_directory
    create_directory(directory_path=output_directory_results)

    output_directory_on = output_directory_results + 'on_' + args.on + '/'
    create_directory(output_directory_on)

    fid_div_calculator = FID_DIVERSITY_CALCULATOR()

    if args.on == 'real':

        GRU_directory = output_directory_results + 'GRU/'
        GRU_trained_on = GRU_directory + 'on_real/'
        GRU_split = GRU_trained_on + 'all/'

        if os.path.exists(GRU_split + 'fid_div.csv'):
            df = pd.read_csv(GRU_split + 'fid_div.csv')
        
        else:
            df = pd.DataFrame(columns=['run','FID','DIVERSITY'])

        runs = get_n_runs(dir=GRU_split)

        for _run in range(runs):

            GRU_run_dir = GRU_split + 'run_' + str(_run) + '/best_model.hdf5'

            fid_trials = []

            for _trial in range(runs): # only because on real samples we have to do more than one trial

                # _trial variable is not used

                x1, x2 = train_test_split(n_X, test_size=0.5, stratify=Y)
                
                fid_trials.append(fid_div_calculator.get_FID(classifier_dir=GRU_run_dir,
                                                             x=x1, y=x2))
            
            div = fid_div_calculator.get_DIVERSITY(classifier_dir=GRU_run_dir,
                                                   x=n_X)
            
            df = df.append({'run' : _run,
                            'FID' : np.mean(fid_trials),
                            'DIVERSITY' : div}, ignore_index=True)
            
            df.to_csv(GRU_split + 'fid_div.csv', index=False)

    elif args.on == 'generated':
    
        factors = [2**i for i in range(args.n_factors+1)]

        GRU_directory = output_directory_results + 'GRU/'
        dataset_dir = GRU_directory + args.dataset + '/'
        GRU_trained_on = dataset_dir + 'on_real/'
        GRU_split = GRU_trained_on + 'all/'

        runs_GRU = get_n_runs(dir=GRU_split)

        gen_models_dir = output_directory_results + 'Generative_models/'
        dataset_dir = gen_models_dir + args.dataset + '/'
        gen_model_dir = dataset_dir + args.generative_model + '/'

        gen_weights = get_dirs(dir=gen_model_dir)

        for gen_weight in gen_weights:

            weights_loss = get_weights_loss(dir=gen_weight)
            print(weights_loss)
            gen_weight_dir = gen_model_dir + gen_weight + '/'
            
            gen_split = gen_weight_dir + 'all/'

            if os.path.exists(gen_split + 'fid_div.csv'):
                df = pd.read_csv(gen_split + 'fid_div.csv')
            
            else:
                df = pd.DataFrame(columns=['run_GRU','run_'+args.generative_model,'generation','factor','FID','DIVERSITY'])
            
            runs_gen = get_n_runs(dir=gen_split)

            for _run_GRU in range(runs_GRU):

                GRU_run_dir = GRU_split + 'run_' + str(_run_GRU) + '/best_model.hdf5'

                for _run_gen in range(runs_gen):

                    gen_model_run = gen_split + 'run_' + str(_run_gen) + '/'

                    for _generation in range(args.n_generations):

                        for factor in factors:

                            if len(df.loc[(df['run_GRU'] == _run_GRU) & (df['run_'+args.generative_model] == _run_gen) & (df['generation'] == _generation) & (df['factor'] == factor)]) > 0:
                                continue
                        
                            if args.generative_model == 'VAE':
                                generator = VAE(output_directory=gen_model_run,
                                                length_TS=length_TS,
                                                n_joints=n_joints,
                                                dim=dim,
                                                n_classes=n_classes,
                                                w_rec=weights_loss['Wrec'],
                                                w_kl=weights_loss['Wkl'])
                            
                            elif args.generative_model == 'SVAE':
                                generator = SVAE(output_directory=gen_model_run,
                                                length_TS=length_TS,
                                                n_joints=n_joints,
                                                dim=dim,
                                                n_classes=n_classes,
                                                w_rec=weights_loss['Wrec'],
                                                w_kl=weights_loss['Wkl'],
                                                w_cls=weights_loss['Wcls'])
                    
                            generated_samples = generator.generate_samples(
                                y_to_generate_from=Y,
                                xtrain=n_X,
                                ytrain=Y,
                                factor=factor,
                                with_labels=False
                            )

                            print("Done generating")

                            fid = fid_div_calculator.get_FID(classifier_dir=GRU_run_dir,
                                                             x=generated_samples,
                                                             y=n_X)
                            
                            div = fid_div_calculator.get_DIVERSITY(classifier_dir=GRU_run_dir,
                                                                   x=generated_samples)
                            
                            df = df.append({
                                'run_GRU' : _run_GRU,
                                'run_' + args.generative_model : _run_gen,
                                'generation' : _generation,
                                'factor' : factor,
                                'FID' : fid,
                                'DIVERSITY' : div}, ignore_index=True)
                            
                            df.to_csv(gen_split + 'fid_div.csv', index=False)
