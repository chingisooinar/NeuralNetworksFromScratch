#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  4 07:18:03 2021

@author: nuvilabs
"""
import numpy as np
class SGD:
    #initialize optimizer - set settings
    #learning rate of 1. is default for this optimizer
    def __init__(self, lr=1.0, decay=0., momentum=0.):
        self.decay = decay
        self.current_lr = lr
        self.lr = lr
        self.iterations = 0
        self.momentum = momentum
    
    #update parameters
    def update_params(self, layer):
        if self.momentum:
            if not hasattr(layer, 'weight_momentums'):
                layer.weight_momentums = np.zeros_like(layer.weights)
                layer.bias_momentums = np.zeros_like(layer.biases)
            
            weight_updates = self.momentum * layer.weight_momentums - self.current_lr * layer.dweights
            layer.weight_momentums = weight_updates
            bias_updates = self.momentum * layer.bias_momentums - self.current_lr * layer.dbiases
            layer.bias_momentums = bias_updates
        else:
            weight_updates = -self.current_lr * layer.dweights
            bias_updates = -self.current_lr * layer.dbiases
        
        layer.weights += weight_updates
        layer.biases += bias_updates
        
    #Call once before any parameter updates
    def pre_update_params(self):
        if self.decay:
            self.current_lr = self.lr * (1. / (1. + self.decay * self.iterations))
            
    #Call once after any parameter updates
    def post_update_params(self):
        self.iterations += 1

class Adagrad:
    #initialize optimizer - set settings
    #learning rate of 1. is default for this optimizer
    def __init__(self, lr=1.0, decay=0., eps=1e-7):
        self.decay = decay
        self.current_lr = lr
        self.lr = lr
        self.iterations = 0
        self.eps = eps
    
    #update parameters
    def update_params(self, layer):
        # If layer does not contain cache arrays,
        # create them filled with zeros
        if not hasattr(layer, 'weight_cache'):
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_cache = np.zeros_like(layer.biases)
        layer.weight_cache += layer.dweights**2
        layer.bias_cache += layer.dbiases**2
        
        layer.weights += -self.lr * layer.dweights / (np.sqrt(layer.weight_cache) + self.eps)
        layer.biases += -self.lr * layer.dbiases / (np.sqrt(layer.bias_cache) + self.eps)
        
        
    #Call once before any parameter updates
    def pre_update_params(self):
        if self.decay:
            self.current_lr = self.lr * (1. / (1. + self.decay * self.iterations))
            
    #Call once after any parameter updates
    def post_update_params(self):
        self.iterations += 1

class RMSprop:
    #initialize optimizer - set settings
    #learning rate of 1. is default for this optimizer
    def __init__(self, lr=1.0, decay=0., eps=1e-7, rho=0.9):
        self.decay = decay
        self.current_lr = lr
        self.lr = lr
        self.iterations = 0
        self.eps = eps
        self.rho = rho
    
    #update parameters
    def update_params(self, layer):
        # If layer does not contain cache arrays,
        # create them filled with zeros
        if not hasattr(layer, 'weight_cache'):
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_cache = np.zeros_like(layer.biases)
        layer.weight_cache = self.rho * layer.weight_cache + (1 - self.rho) * layer.dweights**2
        layer.bias_cache = self.rho * layer.bias_cache + (1 - self.rho) * layer.dbiases**2
        
        layer.weights += -self.lr * layer.dweights / (np.sqrt(layer.weight_cache) + self.eps)
        layer.biases += -self.lr * layer.dbiases / (np.sqrt(layer.bias_cache) + self.eps)
        
        
    #Call once before any parameter updates
    def pre_update_params(self):
        if self.decay:
            self.current_lr = self.lr * (1. / (1. + self.decay * self.iterations))
            
    #Call once after any parameter updates
    def post_update_params(self):
        self.iterations += 1
        
class Adam:
    #initialize optimizer - set settings
    #learning rate of 1. is default for this optimizer
    def __init__(self, lr=1.0, decay=0., eps=1e-7, beta_1=0.9, beta_2=0.999):
        self.decay = decay
        self.current_lr = lr
        self.lr = lr
        self.iterations = 0
        self.eps = eps
        self.beta_1 = beta_1
        self.beta_2 = beta_2
    #update parameters
    def update_params(self, layer):
        # If layer does not contain cache arrays,
        # create them filled with zeros
        if not hasattr(layer, 'weight_cache'):
            layer.weight_momentums = np.zeros_like(layer.weights)
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_momentums = np.zeros_like(layer.biases)
            layer.bias_cache = np.zeros_like(layer.biases)
        # Update momentum with current gradients
        layer.weight_momentums = self.beta_1 * layer.weight_momentums + (1 - self.beta_1) * layer.dweights
        layer.bias_momentums = self.beta_1 * layer.bias_momentums + (1 - self.beta_1) * layer.dbiases
        # Get corrected momentum
        # self.iteration is 0 at first pass
        # and we need to start with 1 here
        weight_momentums_corrected = layer.weight_momentums / (1 - self.beta_1 ** (self.iterations + 1))
        bias_momentums_corrected = layer.bias_momentums / (1 - self.beta_1 ** (self.iterations + 1))
        # Update cache with squared current gradients
        layer.weight_cache = self.beta_2 * layer.weight_cache + (1 - self.beta_2) * layer.dweights**2
        layer.bias_cache = self.beta_2 * layer.bias_cache + (1 - self.beta_2) * layer.dbiases**2
        # Get corrected cache
        weight_cache_corrected = layer.weight_cache / (1 - self.beta_2 ** (self.iterations + 1))
        bias_cache_corrected = layer.bias_cache / (1 - self.beta_2 ** (self.iterations + 1))
        # Vanilla SGD parameter update + normalization
        # with square rooted cache
        layer.weights += -self.lr * weight_momentums_corrected / (np.sqrt(weight_cache_corrected) + self.eps)
        layer.biases += -self.lr * bias_momentums_corrected / (np.sqrt(bias_cache_corrected) + self.eps)
        
        
    #Call once before any parameter updates
    def pre_update_params(self):
        if self.decay:
            self.current_lr = self.lr * (1. / (1. + self.decay * self.iterations))
            
    #Call once after any parameter updates
    def post_update_params(self):
        self.iterations += 1