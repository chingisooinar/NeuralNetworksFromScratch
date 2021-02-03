#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  4 07:18:03 2021

@author: nuvilabs
"""

class SGD:
    #initialize optimizer - set settings
    #learning rate of 1. is default for this optimizer
    def __init__(self, lr=1.0):
        self.lr = lr
    
    #update parameters
    def update_params(self, layer):
        layer.weights += -self.lr * layer.dweight
        layer.biases += -self.lr * layer.dbiases