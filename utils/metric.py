
import numpy as np
import torch

__all__ = ['Statistics']


class Statistics(object):
    """
    Statistics class for computing mean and variance.
    """
    
    def __init__(self):        
        self.sample_mean_sum = 0
        self.sample_var_sum = 0
        self.count = 0
        self.mean = 0
        self.var = 0

    def reset(self):
        self.sample_mean_sum = 0
        self.sample_var_sum = 0
        self.count = 0
        self.mean = 0
        self.var = 0

    def update(self, sample_mean, sample_var):
        self.sample_mean_sum += sample_mean 
        self.sample_var_sum += sample_var
        self.count += 1
        self.mean = self.sample_mean_sum / self.count
        self.var = self.sample_var_sum / self.count
