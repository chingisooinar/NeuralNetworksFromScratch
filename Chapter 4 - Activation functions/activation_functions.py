#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 21 05:47:24 2021

@author: nuvilabs
"""
import numpy as np
class ReLu:
    def __call__(self,inputs):
        return self.forward(inputs)
    def forward(self,inputs):
        return np.maximum(0,inputs)
    
class Softmax:
    def __call__(self,inputs):
        return self.forward(inputs)
    def forward(self,inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis = 1, keepdims = True))
        probs = exp_values/np.sum(exp_values, axis = 1, keepdims= True)
        return probs
    
class Sigmoid:
    def __call__(self,inputs):
        return self.forward(inputs)
    def forward(self,inputs):
        inputs = -inputs
        exp_values = np.exp(inputs - np.max(inputs, axis = 1, keepdims = True))
        probs = 1/(1+exp_values)
        return probs
"""
we also included a subtraction of the largest of the inputs before we did the
exponentiation. There are two main pervasive challenges with neural networks: “dead neurons”
and very large numbers (referred to as “exploding” values). “Dead” neurons and enormous
numbers can wreak havoc down the line and render a network useless over time. The exponential
function used in softmax activation is one of the sources of exploding values.
"""