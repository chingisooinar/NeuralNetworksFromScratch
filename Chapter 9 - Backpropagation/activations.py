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
        self.inputs = inputs
        return np.maximum(0,inputs)
    def backward(self,dvalues):
        self.dinputs = np.copy(dvalues)
        self.dinputs[self.inputs <= 0] = 0
    
class Softmax:
    def __call__(self,inputs):
        return self.forward(inputs)
    def forward(self,inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis = 1, keepdims = True))
        self.output = exp_values/np.sum(exp_values, axis = 1, keepdims= True)
        return self.output
    def backward(self,dvalues):
        """
        ...
        We can perform this operation on each of the Jacobian matrices directly,
        applying the chain rule at the same time (applying the gradient from the loss function) using
        np.dot() — For each sample, it’ll take the row from the Jacobian matrix and multiply it by the
        corresponding value from the loss function’s gradient. As a result, the dot product of each of these
        
        vectors and values will return a singular value, forming a vector of the partial derivatives sample-
        wise and a 2D array (a list of the resulting vectors) batch-wise.
        """
        #S_i_j * sigma_j_k - S_i_j * S_i_k
        self.dinputs = np.empty_like(dvalues)
        
        for index, (single_output,single_dvalues) in enumerate(zip(self.output,dvalues)):
            # Flatten output array
            single_output.reshape_(-1,1)
            # Calculate Jacobian matrix of the output
                                #S_i_j * sigma_j_k      -     S_i_j * S_i_k 
            jacobian_matrix = np.diagflat(single_output) - np.dot(single_output, single_output.T)
            # Calculate sample-wise gradient
            # and add it to the array of sample gradients
            self.dinputs[index] = np.dot(jacobian_matrix, single_dvalues)
class Sigmoid:
    def __call__(self,inputs):
        return self.forward(inputs)
    def forward(self,inputs):
        inputs = -inputs
        exp_values = np.exp(inputs - np.max(inputs, axis = 1, keepdims = True))
        probs = 1/(1+exp_values)
        return probs
