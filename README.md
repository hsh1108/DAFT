# Domain-Aware Fine-Tuning: Enhancing Neural Network Adaptability, AAAI 2024

This repository contains the code for our paper [Domain-Aware Fine-Tuning: Enhancing Neural Network Adaptability](https://arxiv.org/abs/2308.07728) (AAAI 2024). It is built upon the [LP-FT](https://github.com/AnanyaKumar/transfer_learning).

## Batch Normalization Conversion
Our main technique, batch normalization conversion, is easy to implement. You can use the following code to convert batch normalization layers in your model before fine-tuning. You can also find the code in  `utils/transfer.py` file. Statistics class is used to store mean and variance of batch samples. You can find Statistics class in `utils/metric.py` file.

```python
import torch
import torch.nn.functional as F
import copy
from torch.nn.modules.batchnorm import _BatchNorm
from utils.metric import Statistics

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
                    batch_var = torch.var(x, dim=[0, 2, 3])    # it computes sample variance (not population variance)
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
    
```


## Prerequisites
We use link file `disk` to download pretrained models and datasets, because they are too large to download on home directory. `disk` contains `data`, `pretrained_checkpoints`, and `results` folder as follows:
```
disk
├── data                        # Directory to save datasets
├── pretrained_checkpoints      # Directory to save pretrained models
└── results                     # Directory to save results of experiments
```

### Pretrained models
- Download [pretrained models](https://worksheets.codalab.org/bundles/0x57abca4a55ec4f7e9908098beb2633c6) and put them in `disk/pretrained_checkpoints` folder.

### Datasets
Download datasets and put them in `disk/data` folder.
- [DomainNet](http://ai.bu.edu/M3SDA/#dataset)
- Living-17 and Entity : use [BREEDS](https://github.com/MadryLab/BREEDS-Benchmarks/tree/master)
- FMoW : use [WILD](https://wilds.stanford.edu/get_started/)
- [CIFAR-10](http://www.cs.toronto.edu/~kriz/cifar.html), [CIFAR-10.1](https://github.com/modestyachts/CIFAR-10.1), - [CIFAR-10-C](https://zenodo.org/record/2535967), and [STL-10](https://cs.stanford.edu/~acoates/stl10/)


## How to Run
You should change learning rate in yaml file according to dataset and pretrained model. We also provide bash scripts to run our experiments in bash files: `run-moco.sh`, `run-clip.sh`, `run-swav.sh`, and `run-ablation.sh`.

### Linear Probing
Following [LP-FT](https://arxiv.org/pdf/2202.10054.pdf), Logistic Regression Classifier is used for head layer. Logistic Regression use features extracted from pretrained model.


(1) Extract features from pretrained model:
``` 
$ python extract_features.py --config configs/cifar-lp.yaml --run_num 1
```
(2) Train a linear classifier with Logistic Regression:
```
$ python lp_log_reg.py --config configs/cifar-lp.yaml --run_num 1
```


### Fine-tuning
For fine-tuning, we use sgd optimization. 
- FT (Fine-Tuning)
```
python ft_sgd.py --config configs/cifar-ft.yaml --run_num 1
```

- LP-FT (Linear Probe, then Fine-Tuning)
```
python ft_sgd.py --config configs/cifar-lp_ft.yaml--run_num 1
```

### Domain-Aware Fine-Tuning (DAFT)
```
python ft_sgd.py --config configs/cifar-daft.yaml --run_num 1
```
