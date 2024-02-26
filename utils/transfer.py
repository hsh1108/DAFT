import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.modules.batchnorm import _BatchNorm
from utils.metric import Statistics


# function that converts original batchnorm to new batchnorm that computes only running_mean and fix variance.
def convert_bn_params(model, data_loader):
    bn_stats = {}
    
    tmp_model = copy.deepcopy(model)
    tmp_model.eval()
    for name, m in tmp_model.named_modules():
        if isinstance(m, _BatchNorm):
            bn_stats[name] = Statistics()            

            def new_forward(bn, stats_est):
                def lambda_forward(x):
                    x = x.contiguous()     
                    # compute mean of batch samples 
                    batch_mean = torch.mean(x, dim=[0, 2, 3])
                    batch_var = torch.var(x, dim=[0, 2, 3])                 # it computes sample variance (not population variance)
                    stats_est.update(batch_mean.data, batch_var.data)                    

                    # bn forward using calculated mean & var                    
                    return F.batch_norm(
                        x,
                        bn.running_mean,
                        bn.running_var,
                        bn.weight,
                        bn.bias,
                        False,
                        0.0,
                        bn.eps,
                    )
                return lambda_forward
            m.forward = new_forward(m, bn_stats[name])    
    
    print('Computing new mean & var of batchnorm')
    print('Length of Dataloader : ',len(data_loader))
    
    with torch.no_grad():
        for images, _ in data_loader:            
            if torch.cuda.is_available():
                images = images.cuda()                        
            tmp_model(images)                   
       

    print('Converting batchnorm')
    for name, m in model.named_modules():
        if isinstance(m, _BatchNorm):
            # convert weight & bias according to new mean and var without changing result            
            m.bias.data += (bn_stats[name].mean - m.running_mean.data) * m.weight.data / torch.sqrt(m.running_var + m.eps)
            m.weight.data *= torch.sqrt(bn_stats[name].var + m.eps) / torch.sqrt(m.running_var + m.eps)            
            # convert running mean & var
            m.running_mean.data.copy_(bn_stats[name].mean)
            m.running_var.data.copy_(bn_stats[name].var)
    


def normalize(feat, mean, var):
    # feat : [N, C, 1, 1]
    # mean : [1]
    # var : [1]
    feat = torch.reshape(feat, (feat.shape[0], -1))
    if mean is not None:
        feat = feat - mean
    if var is not None:
        feat = feat / torch.sqrt(var)
    return feat
 
def get_new_weight_bias(weight, bias, mean, var):
    # weight : shape = [O, I]
    # bias : shape = [O]
    # mean : shape = [1]
    # var : shape = [1]
    if mean is None and var is None:
        return weight, bias
    new_weight = weight / torch.sqrt(var)    
    new_bias = bias - torch.matmul(new_weight, mean / torch.sqrt(var) * torch.ones_like(new_weight[0]))
    
    return new_weight, new_bias

def convert_head(head, mean, var):
    # head : Linear Layer weight=[O, I], bias=[O]
    # mean : [I]
    # var : [I]
    if mean is None and var is None:
        return head
    new_weight, new_bias = get_new_weight_bias(head.weight, head.bias, mean, var)
    head.weight.data.copy_(new_weight)
    head.bias.data.copy_(new_bias)
    
def unconvert_head(converted_head, mean, var):
    # converted_head : Linear Layer weight=[O, I], bias=[O]
    # mean : [I]
    # var : [I]
    if mean is None and var is None:
        return converted_head
    converted_weight = converted_head.weight    # converted_weight = original_weight / std
    converted_bias = converted_head.bias        # converted_bais = original_bias - original_weight*mean/std
    original_weight = converted_weight * torch.sqrt(var)
    original_bias = converted_bias + torch.matmul(converted_weight, mean / torch.sqrt(var)*torch.ones_like(converted_weight[0]))
    converted_head.weight.data.copy_(original_weight)
    converted_head.bias.data.copy_(original_bias)
    