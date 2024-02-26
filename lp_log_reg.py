# Perform linear probing on extracted features

import numpy as np
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
from scipy.special import softmax
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn import preprocessing
import os
import argparse
import logging
import json
import yaml
import random
import time
import pickle
from quinine import QuinSweep
import utils
import pandas as pd

import copy


log_level = logging.INFO


def normalize_features(features):
    mean = np.mean(features['train'])
    std = np.std(features['train'])
    normalized_features = {}    
    for key in features.keys():
        normalized_features[key] = (features[key] - mean) / std
    return normalized_features

def get_new_weight_bias(coef, intercept, features):
    mean = np.mean(features['train'])  
    std = np.std(features['train']) 
    new_weight = coef / std
    new_bias = intercept - np.matmul(coef, mean / std * np.ones(coef.shape[1]))
    return new_weight, new_bias

def pad_head(weight, bias, classes):
    num_classes = np.max(classes) + 1
    assert num_classes >= len(classes)
    assert len(classes) == weight.shape[0] == bias.shape[0]
    new_weight = np.zeros((num_classes, weight.shape[1]))
    new_bias = np.zeros(num_classes)
    for i, c in enumerate(list(classes)):
        new_weight[c] = weight[i]
        new_bias[c] = bias[i]
    return new_weight, new_bias

def linear_probe(normalized_features, labels,
                 num_cs=100, start_c=-7, end_c=2, max_iter=200):
    Cs=np.logspace(start_c, end_c, num_cs)
    clf=LogisticRegression(max_iter=max_iter, warm_start=True)

    acc_results = []
    best_acc = 0.0
    best_clf, best_coef, best_intercept, best_c = None, None, None, None

    # sweep over Cs and train logistic regression to get best C
    for c in Cs:
        print('Current C =',c)
        clf.set_params(C=c)
        clf.fit(normalized_features['train'], labels['train'])
        # evaluate on test sets
        logging.info('-------------------')
        logging.info('Current C = ' +  str(c))          
        accs = OrderedDict([('C', c)])    
        for key in normalized_features.keys():
            accs[key] = clf.score(normalized_features[key], labels[key])                     
            logging.info('%s acc: %.4f', key, accs[key])
            if key == 'id_val':                
                cur_acc = accs[key]
                if cur_acc > best_acc:
                    best_acc = cur_acc
                    best_clf =  copy.deepcopy(clf)
                    best_coef = copy.deepcopy(clf.coef_)
                    best_intercept = copy.deepcopy(clf.intercept_)
                    best_c = c          
        logging.info('Best id_val acc: %.4f (C = ' + str(best_c) + ')', best_acc)  
        acc_results.append(accs)
    return best_clf, best_coef, best_intercept, best_c, acc_results

def main(config, args):
    # Set save path and feature path
    if args.run_num is not None:
        run_name = 'run_' + args.run_num
    else:
        run_name = 'debug'
    
    save_path = os.path.join(config['save_path'], config['model']['name'] + '-' + config['train_dataset']['name'], 'lp', run_name)
    feature_path = os.path.join(config['save_path'], config['model']['name'] + '-' + config['train_dataset']['name'], 'extracted_features', run_name)
    os.makedirs(save_path, exist_ok=True)
    yaml.safe_dump(config, open(os.path.join(save_path, 'config.yaml'), 'w'))
    utils.setup_logging(save_path, log_level)

    # Check if extracted features exist and load them
    if not os.path.exists(os.path.join(feature_path, 'features.pkl')):
        logging.error('Extracted features not found. Please run extract_features.py first.')
        exit()
    features = pickle.load(open(os.path.join(feature_path, 'features.pkl'), 'rb'))
    labels = pickle.load(open(os.path.join(feature_path, 'labels.pkl'), 'rb'))

    # Normalize features on train set
    if config['transfer']['normalize_features']:
        logging.info('Normalizing features')
        normalized_features = normalize_features(features)
    
        
    # Perform linear probing
    best_clf, best_coef, best_intercept, best_c, acc_results = linear_probe(
        normalized_features, labels,
        num_cs=config['transfer']['reg_sweep_num'],
        start_c=config['transfer']['reg_start'], end_c=config['transfer']['reg_end'],
        max_iter=config['transfer']['max_iter'])

    # reduncancy check
    assert(np.allclose(best_clf.coef_, best_coef))
    assert(np.allclose(best_clf.intercept_, best_intercept))
    
    # Save accuracies
    logging.info('Saving accuracies and weights')
    accs_df = pd.DataFrame(acc_results)
    accs_df.to_csv(os.path.join(save_path, 'results.csv'), sep='\t', index=False)

    # Get new weight and bias
    if config['transfer']['normalize_features']:
        new_weight, new_bias = get_new_weight_bias(best_coef, best_intercept, features)
    else:
        new_weight, new_bias = best_coef, best_intercept

    # Pad the head, if some classes are missing in the training set (e.g. for FMoW)
    if config['transfer']['pad_class']:
        logging.info('Padding classes')
        logging.info(best_clf.classes_)
        new_weight, new_bias = pad_head(new_weight, new_bias, best_clf.classes_)        

    # Save new weight and bias
    pickle.dump((new_weight, new_bias, best_c), open(os.path.join(save_path, 'weights.pkl'), 'wb'))

    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Linear probing')
    parser.add_argument('--config', type=str, metavar='c', help='YAML config', required=True)
    parser.add_argument('--run_num', type=str, default=None, help='experiment number')
    args = parser.parse_args()

    config_sweep = QuinSweep(sweep_config_path=args.config)

    for config in config_sweep:
        print('=====================')
        print(config)
        main(config, args)


