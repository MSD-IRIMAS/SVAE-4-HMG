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

from utils.utils import load_data, normalize_skeletons, create_directory, get_inverse, get_dirs, get_weights_loss, get_n_runs
from classifiers.GRU import GRU_CLASSIFIER
from generators.VAE import VAE
from generators.SVAE import SVAE

def get_args():

    parser = argparse.ArgumentParser(
    description="Choose which samples to train the GRU classifier on with the type of split.")

    parser.add_argument(
        '--train-on',
        help="On which samples to train.",
        choices=['real','generated','augmented'],
        default='real',
        type=str
    )

    parser.add_argument(
        '--dataset',
        help="Which dataset to use.",
        type=str,
        default='HumanAct12'
    )

    parser.add_argument(
        '--factor',
        help="Factor to generate number of samples = factor * number of real samples.",
        type=int,
        default=1
    )

    parser.add_argument(
        '--generative-model',
        help="Which generative model to use if training on generated or augmented.",
        type=str,
        choices=['VAE','SVAE'],
        default='VAE',
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
        default='train_test'
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
        '--test-size',
        help="Percentage of test size if the split chosen is train_test.",
        type=float,
        default=0.25
    )

    parser.add_argument(
        '--n-generations',
        help="Number of generations to do per generative model pre-trained.",
        type=int,
        default=5
    )

    parser.add_argument(
        '--n-epochs',
        type=int,
        default=2000
    )

    args = parser.parse_args()

    if args.factor < 1 and args.train_on != 'real':
        raise ValueError("Factor can not be less than 1.")
    
    if (args.test_size > 1 or args.test_size <= 0) and args.split == 'train_test':
        raise ValueError("Test size is a percentage, should be between 0 and 1.")

    return args
    
if __name__ == "__main__":

    args = get_args()

    output_directory_results = args.output_directory
    create_directory(output_directory_results)

    output_directory_clf = output_directory_results + 'GRU/'
    create_directory(output_directory_clf)

    output_directory_dataset = output_directory_clf + args.dataset + '/'
    create_directory(output_directory_dataset)

    output_directory_train_on = output_directory_dataset + 'on_' + args.train_on + '/'
    create_directory(output_directory_train_on)

    output_directory_split = output_directory_train_on + args.split + '/'
    create_directory(output_directory_split)

    dataset_dir = 'datasets/' + args.dataset + '/'
    X, Y, S = load_data(root_dir=dataset_dir)

    le = LE()
    Y = le.fit_transform(Y)

    le = LE()
    S = le.fit_transform(S)

    if args.split == 'cross_subject':
        if args.subject < S.min() or args.subject > S.max():
            raise ValueError("Wrong subject, does not exist.")
    
    length_TS = int(X.shape[1])
    n_joints = int(X.shape[2])
    dim = int(X.shape[3])

    if args.train_on == 'real':

        if args.split == 'all':

            if os.path.exists(output_directory_split + 'results.csv'):
                df_results = pd.read_csv(output_directory_split + 'results.csv')
        
            else:
                df_results = pd.DataFrame(columns=['run_GRU','Accuracy'])

            n_classes = len(np.unique(Y))

            for _run_GRU in range(args.runs):

                if len(df_results.loc[(df_results['run_GRU'] == _run_GRU)]) > 0:
                    continue

                output_directory_run = output_directory_split + 'run_' + str(_run_GRU) + '/'
                create_directory(output_directory_run)

                n_X, min_X, max_X, min_Y, max_Y, min_Z, max_Z = normalize_skeletons(X=X)

                clf = GRU_CLASSIFIER(output_directory=output_directory_run, length_TS=length_TS, n_joints=n_joints,
                                     dim=dim, n_classes=n_classes, n_epochs=args.n_epochs)

                clf.fit(xtrain=n_X, ytrain=Y)
                acc = clf.predict(xtest=n_X, ytest=Y)

                df_results = df_results.append({
                    'run_GRU' : _run_GRU,
                    'Accuracy' : acc}, ignore_index=True)
                
                df_results.to_csv(output_directory_split + 'results.csv', index=False)
        
        elif args.split == 'train_test':

            if os.path.exists(output_directory_split + 'results.csv'):
                df_results = pd.read_csv(output_directory_split + 'results.csv')
        
            else:
                df_results = pd.DataFrame(columns=['split','run_GRU','Accuracy'])

            leave_out_splits = np.load('datasets/' + args.dataset + '/leave_out.npy')

            for _split, leave_out_split in enumerate(leave_out_splits):

                if len(df_results.loc[(df_results['split'] == _split)]) == args.runs:
                    continue

                output_directory_n_split = output_directory_split + 'split_' + str(_split) + '/'
                create_directory(output_directory_n_split)

                x_train = X.copy()
                ytrain = Y.copy()
                S_train = S.copy()

                x_test = X.copy()
                ytest = Y.copy()
                S_test = S.copy()

                for s in leave_out_split:
                    
                    x_train = x_train[S_train != s]
                    ytrain = ytrain[S_train != s]
                    S_train = S_train[S_train != s]
                
                for s in get_inverse(S=S, leave_out_split=leave_out_split):

                    x_test = x_test[S_test != s]
                    ytest = ytest[S_test != s]
                    S_test = S_test[S_test != s]

                n_classes = len(np.unique(ytrain))

                xtrain, min_X, max_X, min_Y, max_Y, min_Z, max_Z = normalize_skeletons(X=x_train)
                xtest, _,_, _,_, _,_ = normalize_skeletons(X=x_test,
                                                           min_X=min_X, max_X=max_X,
                                                           min_Y=min_Y, max_Y=max_Y,
                                                           min_Z=min_Z, max_Z=max_Z)
                
                n_classes = len(np.unique(ytrain))

                for _run_GRU in range(args.runs):

                    output_directory_run = output_directory_n_split + 'run_' + str(_run_GRU) + '/'
                    create_directory(output_directory_run)

                    clf = GRU_CLASSIFIER(output_directory=output_directory_run, length_TS=length_TS, n_joints=n_joints,
                                         dim=dim, n_classes=n_classes, n_epochs=args.n_epochs)

                    if not os.path.exists(output_directory_run + 'loss.pdf'):
                        clf.fit(xtrain=xtrain, ytrain=ytrain, xval=xtest, yval=ytest, plot_test=True)

                    acc = clf.predict(xtest=xtest, ytest=ytest)

                    df_results = df_results.append({
                        'split' : _split,
                        '_run_GRU' : _run_GRU,
                        'Accuracy' : acc}, ignore_index=True)
                    
                    df_results.to_csv(output_directory_split + 'results.csv', index=False)
    
    elif args.train_on == 'generated':

        if args.split == 'train_test':

            output_directory_gen_model = output_directory_split + args.generative_model + '/'
            create_directory(output_directory_gen_model)

            leave_out_splits = np.load('datasets/' + args.dataset + '/leave_out.npy')

            gen_models_dir = output_directory_results + 'Generative_models/'
            gen_model_dir = gen_models_dir + args.generative_model + '/'
            
            if args.generative_model != 'SVAE':
                    
                    gen_weight_dir = gen_model_dir + 'Wrec_' + str(args.weight_rec) + '_Wkl_' + str(args.weight_kl) + '/'
                    output_directory_gen_weights = output_directory_gen_model +  'Wrec_' + str(args.weight_rec) + '_Wkl_' + str(args.weight_kl) + '/'

            else:

                gen_weight_dir = gen_model_dir + 'Wrec_' + str(args.weight_rec) + '_Wkl_' + str(args.weight_kl) + '_Wcls_' + str(args.weight_cls) + '/'
                output_directory_gen_weights = output_directory_gen_model + 'Wrec_' + str(args.weight_rec) + '_Wkl_' + str(args.weight_kl) + '_Wcls_' + str(args.weight_cls) + '/'

            create_directory(output_directory_gen_weights)

            weights_loss = {
                'Wrec' : args.weight_rec,
                'Wkl' : args.weight_kl,
                'Wcls' : args.weight_cls
            }

            gen_split = gen_weight_dir + 'train_test/'

            if os.path.exists(output_directory_gen_weights + 'results.csv'):
                df_results = pd.read_csv(output_directory_gen_weights + 'results.csv')
        
            else:
                df_results = pd.DataFrame(columns=['split','run_'+args.generative_model,'generation','run_GRU','Accuracy'])

            for _split, leave_out_split in enumerate(leave_out_splits):

                gen_n_split = gen_split + 'split_' + str(_split) + '/'
                runs_gen = get_n_runs(dir=gen_n_split)

                x_train = X.copy()
                ytrain = Y.copy()
                S_train = S.copy()

                for s in leave_out_split:

                    x_train = x_train[S_train != s]
                    ytrain = ytrain[S_train != s]
                    S_train = S_train[S_train != s]

                x_test = X.copy()
                ytest = Y.copy()
                S_test = S.copy()

                for s in get_inverse(S=S, leave_out_split=leave_out_split):

                    x_test = x_test[S_test != s]
                    ytest = ytest[S_test != s]
                    S_test = S_test[S_test != s]
                
                xtrain, min_X,max_X, min_Y,max_Y, min_Z,max_Z = normalize_skeletons(X=x_train)
                xtest, _,_, _,_, _,_ = normalize_skeletons(X=x_test, min_X=min_X,max_X=max_X,
                                                        min_Y=min_Y,max_Y=max_Y,
                                                        min_Z=min_Z,max_Z=max_Z)

                n_classes = len(np.unique(ytrain))

                for _run_gen in range(runs_gen):

                    output_directory_gen_run = output_directory_gen_weights + 'run_gen_' + str(_run_gen) + '/'
                    create_directory(output_directory_gen_run)

                    gen_model_run = gen_n_split + 'run_' + str(_run_gen) + '/'

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
                    
                    for _generation in range(args.n_generations):

                        if len(df_results.loc[(df_results['split'] == _split) & (df_results['run_'+args.generative_model] == _run_gen) & (df_results['generation'] == _generation)]) == args.runs:
                            continue

                        output_directory_generation = output_directory_gen_run + 'generation_' + str(_generation) + '/'
                        create_directory(output_directory_generation)

                        generated_samples, labels_generated = generator.generate_samples(
                            y_to_generate_from=ytrain,
                            xtrain=xtrain, ytrain=ytrain,
                            factor=args.factor,
                            with_labels=True)
                        
                        for _run_GRU in range(args.runs):

                            output_directory_run = output_directory_generation + 'run_' + str(_run_GRU) + '/'
                            create_directory(output_directory_run)

                            clf = GRU_CLASSIFIER(output_directory=output_directory_run,
                                                    length_TS=length_TS,
                                                    n_joints=n_joints,
                                                    dim=dim,
                                                    n_classes=n_classes,
                                                    n_epochs=args.n_epochs)

                            clf.fit(xtrain=generated_samples,ytrain=labels_generated,
                                    xval=xtest, yval=ytest, plot_test=True)
                            acc = clf.predict(xtest=xtest, ytest=ytest)
                
                            df_results = df_results.append({
                                'split' : _split,
                                'run_'+args.generative_model : _run_gen,
                                'generation' : _generation,
                                'run_GRU' : _run_GRU,
                                'Accuracy' : acc}, ignore_index=True)

                            df_results.to_csv(output_directory_gen_weights + 'results.csv', index=False)
    
    elif args.train_on == 'augmented':

        if args.split == 'train_test':

            output_directory_gen_model = output_directory_split + args.generative_model + '/'
            create_directory(output_directory_gen_model)

            leave_out_splits = np.load('datasets/' + args.dataset + '/leave_out.npy')

            gen_models_dir = output_directory_results + 'Generative_models/'
            gen_model_dir = gen_models_dir + args.generative_model + '/'

            if args.generative_model != 'SVAE':
                    
                    gen_weight_dir = gen_model_dir + 'Wrec_' + str(args.weight_rec) + '_Wkl_' + str(args.weight_kl) + '/'
                    output_directory_gen_weights = output_directory_gen_model +  'Wrec_' + str(args.weight_rec) + '_Wkl_' + str(args.weight_kl) + '/'

            else:

                gen_weight_dir = gen_model_dir + 'Wrec_' + str(args.weight_rec) + '_Wkl_' + str(args.weight_kl) + '_Wcls_' + str(args.weight_cls) + '/'
                output_directory_gen_weights = output_directory_gen_model + 'Wrec_' + str(args.weight_rec) + '_Wkl_' + str(args.weight_kl) + '_Wcls_' + str(args.weight_cls) + '/'

            create_directory(output_directory_gen_weights)

            weights_loss = {
                'Wrec' : args.weight_rec,
                'Wkl' : args.weight_kl,
                'Wcls' : args.weight_cls
            }

            gen_split = gen_weight_dir + 'train_test/'

            if os.path.exists(output_directory_gen_weights + 'results.csv'):
                df_results = pd.read_csv(output_directory_gen_weights + 'results.csv')
        
            else:
                df_results = pd.DataFrame(columns=['split','run_'+args.generative_model,'generation','run_GRU','Accuracy'])

            for _split, leave_out_split in enumerate(leave_out_splits):

                gen_n_split = gen_split + 'split_' + str(_split) + '/'

                output_directory_split = output_directory_gen_weights + 'split_' + str(_split) + '/'
                create_directory(output_directory_split)

                runs_gen = get_n_runs(dir=gen_n_split)

                x_train = X.copy()
                ytrain = Y.copy()
                S_train = S.copy()

                for s in leave_out_split:

                    x_train = x_train[S_train != s]
                    ytrain = ytrain[S_train != s]
                    S_train = S_train[S_train != s]

                x_test = X.copy()
                ytest = Y.copy()
                S_test = S.copy()

                for s in get_inverse(S=S, leave_out_split=leave_out_split):

                    x_test = x_test[S_test != s]
                    ytest = ytest[S_test != s]
                    S_test = S_test[S_test != s]

                xtrain, min_X,max_X, min_Y,max_Y, min_Z,max_Z = normalize_skeletons(X=x_train)
                xtest, _,_, _,_, _,_ = normalize_skeletons(X=x_test, min_X=min_X,max_X=max_X,
                                                        min_Y=min_Y,max_Y=max_Y,
                                                        min_Z=min_Z,max_Z=max_Z)
                
                n_classes = len(np.unique(ytrain))

                for _run_gen in range(runs_gen):

                    output_directory_gen_run = output_directory_split + 'run_gen_' + str(_run_gen) + '/'
                    create_directory(output_directory_gen_run)

                    gen_model_run = gen_n_split + 'run_' + str(_run_gen) + '/'

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

                    for _generation in range(args.n_generations):

                        if len(df_results.loc[(df_results['split'] == _split) & (df_results['run_'+args.generative_model] == _run_gen) & (df_results['generation'] == _generation)]) == args.runs:
                            continue

                        output_directory_generation = output_directory_gen_run + 'generation_' + str(_generation) + '/'
                        create_directory(output_directory_generation)

                        # generated_samples, labels_generated = generator.generate_samples(
                        #     y_to_generate_from=ytrain,
                        #     xtrain=xtrain, ytrain=ytrain,
                        #     factor=args.factor,
                        #     with_labels=True)

                        generated_samples, labels_generated = generator.generate_samples_to_fix_labeling(xtrain=xtrain, ytrain=ytrain)
                        
                        for _run_GRU in range(args.runs):

                            output_directory_run = output_directory_generation + 'run_' + str(_run_GRU) + '/'
                            create_directory(output_directory_run)

                            clf = GRU_CLASSIFIER(output_directory=output_directory_run,
                                                    length_TS=length_TS,
                                                    n_joints=n_joints,
                                                    dim=dim,
                                                    n_classes=n_classes,
                                                    n_epochs=args.n_epochs)
                            
                            x_augmentation = np.concatenate((xtrain, generated_samples), axis=0)
                            y_augmentation = np.concatenate((ytrain, labels_generated), axis=0)

                            clf.fit(xtrain=x_augmentation,ytrain=y_augmentation,
                                    xval=xtest, yval=ytest, plot_test=True)
                            
                            acc = clf.predict(xtest=xtest, ytest=ytest)
                
                            df_results = df_results.append({
                                'split' : _split,
                                'run_'+args.generative_model : _run_gen,
                                'generation' : _generation,
                                'run_GRU' : _run_GRU,
                                'Accuracy' : acc}, ignore_index=True)

                            df_results.to_csv(output_directory_gen_weights + 'results.csv', index=False)