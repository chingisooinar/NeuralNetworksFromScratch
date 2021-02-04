#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  4 07:18:03 2021

@author: nuvilabs
"""

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