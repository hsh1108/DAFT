# Thanks to Michael Xie for inital version of these utilities.

import ast
from copy import deepcopy
import json
import logging
import importlib
import inspect
import numpy as np
import torch



#######################################################################
# update configs
non_sgd_optimizer_names = [
    'torch.optim.Adam', 'torch.optim.AdamW', 'torch.optim.RMSprop', 'torch_optimizer.Lamb',
    'torch_optimizer.lamb.Lamb'
]
def update_optimizer_args(config):    
    # If the optimizer is not SGD, remove the momentum and nesterov args.    
    if config['optimizer']['classname'] in non_sgd_optimizer_names:         
        del config['optimizer']['args']['momentum']
        del config['optimizer']['args']['nesterov']


#######################################################################
# set random seed
def set_random_seed(seed):
    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)


#######################################################################
# logging
def setup_logging(log_dir, level=logging.DEBUG):
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.getLogger().setLevel(logging.INFO)
    logger = logging.getLogger('')
    logger.handlers = []
    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
        datefmt='%Y-%m-%d:%H:%M:%S', level=level,
        filename=log_dir+'/logs.txt')
    






#######################################################################


def count_parameters(model, trainable):
    return sum(p.numel() for p in model.parameters() if p.requires_grad == trainable)

def save_json(save_path, save_dict):
    with open(save_path, 'w') as outfile:
        json.dump(save_dict, outfile)



    if hasattr(m, "trainable_params"):
        # "trainable_params" is custom module function
        return m.trainable_params()
    return m.parameters()

def save_ckp(epoch, model, optimizer, scheduler, model_dir, chkpt_name):
    checkpoint_fpath = str(model_dir / chkpt_name)
    logging.info(f"Saving to checkpoint {checkpoint_fpath}")
    state = {
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict()
    }
    torch.save(state, checkpoint_fpath)

def load_ckp(checkpoint_fpath, model, optimizer=None, scheduler=None, reset_optimizer=False):
    logging.info(f"Loading from checkpoint {checkpoint_fpath}")
    checkpoint = torch.load(checkpoint_fpath)
    model.load_state_dict(checkpoint['state_dict'], strict=False)
    epoch = 0
    if not reset_optimizer:
        if 'optimizer' in checkpoint and optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer'])
        if 'scheduler' in checkpoint and scheduler is not None:
            scheduler.load_state_dict(checkpoint['scheduler'])
        if 'epoch' in checkpoint:
            epoch = checkpoint['epoch']
        else:
            epoch = int(checkpoint_fpath.split('epoch')[1].split('.')[0])
    return epoch

def update_config(unparsed, config):
    # handle unknown arguments that change yaml config components
    # Embedded arguments e.g. loss.args must already exist in config to be updated.
    for unparsed_option in unparsed:
        option_name, val = unparsed_option.split('=')
        # get rid of --
        option_name = option_name[2:].strip()
        # handle nesting
        option_name_list = option_name.split('.')

        # interpret the string as int, float, string, bool, etc
        try:
            val = ast.literal_eval(val.strip())
        except Exception:
            # keep as string
            val = val.strip()

        curr_dict = config
        for k in option_name_list[:-1]:
            try:
                curr_dict = curr_dict.get(k)
            except:
                raise ValueError(f"Dynamic argparse failed: Keys: {option_name_list} Dict: {config}")
        curr_dict[option_name_list[-1]] = val
    return config


def to_device(obj, device):
    '''
    Wrapper around torch.Tensor.to that handles the case when obj is a
    container.
    Parameters
    ----------
    obj : Union[torch.Tensor, List[torch.Tensor], Dict[Any, Any]]
        Object to move to the specified device.
    device : str
        Describes device to move to.
    Returns
    -------
    Same type as obj.
    '''
    if isinstance(obj, list):
        return [item.to(device) for item in obj]
    elif isinstance(obj, dict):
        res = {}
        for key in obj:
            value = obj[key]
            if isinstance(value, torch.Tensor):
                value = value.to(device)
            res[key] = value
        return res
    else:
        return obj.to(device)


def set_requires_grad(component, val):
    for param in component.parameters():
        param.requires_grad = val

