from quinine import QuinSweep

import argparse
import os, sys
import datetime
import logging
import numpy as np
import json
import random
import time
import torch
import yaml
import torch.backends.cudnn as cudnn
import utils
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import pandas as pd
import copy


log_level = logging.INFO

def train_epoch(config, results, net, train_loader, optimizer, criterion):	
	
	net.get_feature_extractor().eval()
	net.get_last_layer().train()
	
	batch_idx = 0
	correct = 0
	total = 0
	loss_sum = 0
	for (image, label)	in train_loader:
		batch_idx += 1
		image = image.cuda()
		label = label.cuda()
		output = net(image)
		loss = criterion(output, label)
		loss_sum += loss.item()
		_, predicted = output.max(1)
		total += label.size(0)
		correct += predicted.eq(label).sum().item()

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()		
		if batch_idx % config['log_interval'] == 0:
			logging.info('Train loss: %.3f', loss.item())
	results['train_acc'].append(100. * correct / total)
	results['train_loss'].append(loss_sum/(batch_idx+1))
	return results

def test_epoch(config, results, net, test_loaders, criterion, max_test_examples):
	net.eval()	
	with torch.no_grad():
		for name, test_loader in test_loaders.items():			
			correct = 0
			total = 0
			loss_sum = 0	
			logging.info('-------------------')		
			logging.info('Testing on %s', name)
			for batch_idx, (image, label) in enumerate(test_loader):
				image = image.cuda()
				label = label.cuda()				
				output = net(image)
				loss = criterion(output, label)
				loss_sum += loss.item()
				_, predicted = output.max(1)
				total += label.size(0)
				correct += predicted.eq(label).sum().item()
				if batch_idx * config['batch_size'] >= max_test_examples[name]:
					break				
			results['{}_acc'.format(name)].append(100. * correct / total)
			results['{}_loss'.format(name)].append(loss_sum/(batch_idx+1))
			best_ind_on_id = np.argmax(results['id_val_acc'])	
			best_acc_on_id = results['{}_acc'.format(name)][best_ind_on_id]	
			best_ind_on_ood = np.argmax(results['{}_acc'.format(name)])
			best_acc_on_ood = results['{}_acc'.format(name)][best_ind_on_ood] 
			logging.info(
				'Loss: %.3f | Acc: %.3f%% (%d/%d) | Best Acc on ID : %.3f%% (epoch %d) | Best Acc on %s :  %.3f%% (epoch %d)', 
				loss_sum/(batch_idx+1), 100.*correct/total, correct, total, 
				best_acc_on_id, best_ind_on_id+1, 
				name, best_acc_on_ood, best_ind_on_ood+1)
	return results



def main(config, args):	
	# Set random seed	
	config['seed']=args.seed
	utils.set_random_seed(config['seed']) 

	# Set path to save experimental results	
	if args.run_num is not None:
		run_name = 'run_' + args.run_num
	else:
		run_name = 'debug'
	
	exp_name = 'Head_LR' + str(config['optimizer']['args']['lr'])
	save_path = os.path.join(config['save_path'], config['model']['name'] + '-' + config['train_dataset']['name'], config['transfer']['name'], exp_name, run_name)

	os.makedirs(save_path, exist_ok=True)	
	yaml.safe_dump(config, open(os.path.join(save_path, "config.yaml"), 'w'), sort_keys=False)
	utils.setup_logging(save_path, log_level)

	# Update optimizer args
	utils.update_optimizer_args(config)	
	
	print(args)                                                                                                          
	print(config)

	# Get data loaders
	train_loader, sub_train_loader = utils.get_train_loader(config)
	test_loaders, max_test_examples = utils.get_test_loaders(config)
	
	# Build model
	net = utils.build_model(config) 	
	net.new_last_layer(config['num_classes'])	

	if config['transfer']['use_lp_ft_model']:		
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
		

	if torch.cuda.is_available():
		logging.info('Using GPU')
		net = net.cuda()
		cudnn.benchmark = True
					
		
	net_initial = copy.deepcopy(net)
	net_initial.eval()

	# loss function
	criterion = utils.initialize(config['criterion'])

	# get optimizer and scheduler				
	params_head = net.get_last_layer().parameters()
	param_groups = [        
		{'params': params_head, 'lr': config['optimizer']['args']['lr']}
    ]   
	optimizer = utils.initialize(config['optimizer'], update_args={'params': param_groups})
	scheduler = utils.initialize(config['scheduler'], update_args={'optimizer': optimizer})

	# load checkpoint if resuming		
	if args.resume:
		checkpoint = torch.load(os.path.join(save_path, 'checkpoint.pt'))		
		net.load_state_dict(checkpoint['net'])
		optimizer.load_state_dict(checkpoint['optimizer'])
		scheduler.load_state_dict(checkpoint['scheduler'])
		start_epoch = checkpoint['current_epoch']					
		best_acc = checkpoint['best_acc']
		results = checkpoint['results']
		logging.info('Resuming from epoch %d', start_epoch)
	else:
		start_epoch = 0
		best_acc = 0
		results = {}
		results['train_acc'] = []
		results['train_loss'] = []
		for name, _ in test_loaders.items():
			results['{}_acc'.format(name)] = []
			results['{}_loss'.format(name)] = []			
	
	if config['transfer']['use_lp_ft_model']:		
		logging.info('====================================================')
		logging.info('transfer learning method: %s',  config['transfer']['name'])
		logging.info('Number of epochs for transfer learning: %d', config['transfer']['epochs'])
		logging.info('Evaluate before start training')
		eval_results = {}
		eval_results['train_acc'] = []
		eval_results['train_loss'] = []
		for name, _ in test_loaders.items():
			print(name)
			eval_results['{}_acc'.format(name)] = []
			eval_results['{}_loss'.format(name)] = []			
		eval_results = test_epoch(config, eval_results, net, test_loaders, criterion, max_test_examples)			
	
	
	# train and evaluate	
	for e in range(start_epoch, config['transfer']['epochs']):	
		logging.info('====================================================')
		logging.info('Current epoch %d (LR= %f)', e+1,  optimizer.param_groups[0]['lr'])
		results = train_epoch(config, results, net, train_loader, optimizer, criterion)
		results = test_epoch(config, results, net, test_loaders, criterion, max_test_examples)				
		acc = results['id_val_acc'][e]
		scheduler.step()		
		if acc > best_acc:
			best_acc = acc
			logging.info('Saving checkpoint')
			torch.save(
				{
				'current_epoch': e + 1,
				'net': net.state_dict(),
				'best_acc': best_acc,
				'optimizer' : optimizer.state_dict(),
				'scheduler' : scheduler.state_dict(),
				'results' : results,
				},
			os.path.join(save_path, 'checkpoint.pt'))
			utils.analyze_networks(net, net_initial, test_loaders['id_val'], save_path)

	# Save accuracies
	logging.info('Saving accuracies and weights')
	accs_df = pd.DataFrame(results)
	accs_df.to_csv(os.path.join(save_path, 'results.csv'), sep='\t', index=False)

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--config', type=str, metavar='c',
                        help='YAML config', required=True)
	parser.add_argument('--resume', action='store_true')
	parser.add_argument('--seed', type=int, default=None, help='random seed')
	parser.add_argument('--run_num', type=str, default=None, help='experiment number')
	
	args = parser.parse_args()	

	config_sweep = QuinSweep(sweep_config_path=args.config)

	
	for config in config_sweep:
		print('=====================')
		main(config, args)


