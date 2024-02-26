# Extract features an labels, save them to a pickle file
from quinine import QuinSweep

import os
import argparse
import json
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import utils
import quinine
import pickle

def get_features_labels(net, data_loader):
    net.eval()
    features = []
    labels = []
    with torch.no_grad():
        for batch_idx, (image, label) in enumerate(data_loader):
            if torch.cuda.is_available():
                image = image.cuda()
                label = label.cuda()
            output = net.get_features(image)
            features.append(output.cpu().numpy())
            labels.append(label.cpu().numpy())
            
    features = np.squeeze(np.concatenate(features, axis=0))
    labels = np.concatenate(labels, axis=0)
    return features, labels

def main(config, args):    	
    # Set save path
    if args.run_num is not None:
        run_name = 'run_' + args.run_num
    else:
        run_name = 'debug'
    
    # Build model
    net = utils.build_model(config) 	

    """
    if config['transfer']['use_lp_ft_model']:		
        net.new_last_layer(config['num_classes'])	
        pretrained_model = os.path.join(
            config['save_path'],
            config['model']['name'] + '-' + config['train_dataset']['name'],
            'lp_ft',
            'best',
            run_name
            )	
        if not os.path.exists(pretrained_model):
            raise ValueError('run_num: {} is not found in lp_ft folder.'.format(run_name))
        
        if os.path.exists(os.path.join(pretrained_model, 'checkpoint.pt')):
            checkpoint = torch.load(os.path.join(pretrained_model, 'checkpoint.pt'))
            net.load_state_dict(checkpoint['net'])
        else:
            raise ValueError('No pretrained model found')        
            
        save_path = os.path.join(config['save_path'], config['model']['name'] + '-' + config['train_dataset']['name'], 'lp_ft_extracted_features', run_name)    
        os.makedirs(save_path, exist_ok=True)

    else:           
    """ 
    save_path = os.path.join(config['save_path'], config['model']['name'] + '-' + config['train_dataset']['name'], 'extracted_features', run_name)    
    os.makedirs(save_path, exist_ok=True)

                

    if torch.cuda.is_available():
        print('Using GPU')
        net = net.cuda()
        cudnn.benchmark = True
    

    # Get data loaders
    train_loader, _ = utils.get_train_loader(config)
    test_loaders, _ = utils.get_test_loaders(config)
    
    features = {}
    labels = {}
    
    # Extract train and test features
    print('Extracting features and labels for train set.')
    train_features, train_labels = get_features_labels(net, train_loader)
    features['train'] = train_features
    labels['train'] = train_labels
    
    for name, test_loader in test_loaders.items():
        print('Extracting features and labels for test set: {}'.format(name))
        test_features, test_labels = get_features_labels(net, test_loader)
        features[name] = test_features
        labels[name] = test_labels
        
    
    # Save features and labels
    print('Saving features and labels to pickle file')
    with open(os.path.join(save_path, 'features.pkl'), 'wb') as f:
        pickle.dump(features, f)
    with open(os.path.join(save_path, 'labels.pkl'), 'wb') as f:
        pickle.dump(labels, f)
    print('Finished.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract features')
    parser.add_argument('--config', type=str, metavar='c', help='YAML config', required=True)
    parser.add_argument('--run_num', type=str, default=None, help='experiment number')
    args = parser.parse_args()	

    config_sweep = QuinSweep(sweep_config_path=args.config)

    for config in config_sweep:
        print('=====================')
        main(config, args)


