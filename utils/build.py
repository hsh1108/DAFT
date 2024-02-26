import logging
import importlib
import inspect
import torch
import numpy as np

from copy import deepcopy
from torchvision import transforms

"""
 Setup model and data loaders for train and test.
"""

#######################################################################
# Initialize a class from a config dict.
def initialize(obj_config, update_args=None):
    classname = obj_config['classname']
    kwargs = obj_config.get('args')
    if kwargs is None:
        kwargs = {}
    if update_args is not None:
        kwargs.update(update_args)
    return initialize_obj(classname, kwargs)

def initialize_obj(classname, args_dict=None):
    module_name, class_name = classname.rsplit(".", 1)
    Class = getattr(importlib.import_module(module_name), class_name)
    if not inspect.isclass(Class):
        raise ValueError("Can only initialize classes, are you passing in a function?")
    # filter by argnames
    if args_dict is not None:
        argspec = inspect.getfullargspec(Class.__init__)
        argnames = argspec.args
        for k, v in args_dict.items():
            if k not in argnames:
                raise ValueError(f"{k}, {v} not found in {Class}")
        args_dict = {k: v for k, v in args_dict.items()
                     if k in argnames}
        defaults = argspec.defaults
        # add defaults
        if defaults is not None:
            for argname, default in zip(argnames[-len(defaults):], defaults):
                if argname not in args_dict:
                    args_dict[argname] = default
        class_instance = Class(**args_dict)
    else:
        class_instance = Class()
    return class_instance

#######################################################################
# Model
def build_model(config):
    net = initialize(config['model'])    
    return net

#######################################################################
# Data
def init_dataset(dataset_config, transform_config):
    transform = init_transform(transform_config) 
    dataset_config_copy = deepcopy(dataset_config)
    dataset_kwargs = {'transform': transform}
    dataset = initialize(dataset_config_copy, dataset_kwargs)
    return dataset

def init_transform(config_transforms):
    transform_list = [initialize(trans) for trans in config_transforms]
    return transforms.Compose(transform_list)

def get_train_loader(config, shuffle=True):
    train_data = init_dataset(config['train_dataset'], config['train_transforms'])    
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=config['batch_size'],
        shuffle=shuffle, num_workers=config['num_workers'])            

    sub_train_data = deepcopy(train_data)  
    if len(sub_train_data) > 16000:  
        rand_indexes = torch.randperm(len(sub_train_data)).tolist()[:16000]
        sub_train_data = torch.utils.data.Subset(sub_train_data, rand_indexes)    
        
    sub_train_loader = torch.utils.data.DataLoader(
        sub_train_data, batch_size=config['batch_size'],
        shuffle=shuffle, num_workers=config['num_workers'])            
    return train_loader, sub_train_loader


def get_test_loaders(config, shuffle=False):
    test_loaders = {}
    max_test_examples = {}
    logging.info('Found %d testing datasets.', len(config['test_datasets']))
    for test_dataset_config in config['test_datasets']:
        logging.info('test dataset config: ' + str(test_dataset_config))
        # Initialize dataset and data loader.
        # Shuffle is True in case we only test part of the test set.
        test_data = init_dataset(test_dataset_config, config['test_transforms'])
        test_loader = torch.utils.data.DataLoader(
            test_data, batch_size=config['batch_size'],
            shuffle=shuffle, num_workers=config['num_workers'])
        test_config_name = test_dataset_config['name']
        test_loaders[test_config_name] = test_loader
        # Some test datasets like CINIC are huge so we only test part of the dataset.
        if 'max_test_examples' in test_dataset_config:
            logging.info(
                'Only logging %d examples for %s', test_dataset_config['max_test_examples'],
                test_dataset_config['name'])
            max_test_examples[test_config_name] = test_dataset_config['max_test_examples']
        else:
            max_test_examples[test_config_name] = float('infinity')
        logging.info('test loader name: ' + test_dataset_config['name'])
        logging.info('test loader: ' + str(test_loader))
        logging.info('test transform: ' + str(config['test_transforms']))
    return test_loaders, max_test_examples

