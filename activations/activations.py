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
    def forward(self,inputs, training):
        self.inputs = inputs
        self.output =  np.maximum(0,inputs)
    def backward(self,dvalues):
        self.dinputs = np.copy(dvalues)
        self.dinputs[self.inputs <= 0] = 0
    # Calculate predictions for outputs
    def predictions(self, outputs):
        return outputs
class Tanh:
    """
    Hyperbolic Tangent Function(Tanh).
    Implement forward & backward path of Tanh.
    """
    def __init__(self):
        self.out = None
        
    def __call__(self,inputs):
        return self.forward(inputs)
    
    def forward(self, z):
        """
        Hyperbolic Tangent Forward.

        z --> (Tanh) --> self.out

        [Inputs]
            z : Tanh input in any shape.

        [Outputs]
            self.out : Values applied elementwise tanh function on input 'z'.

        """
        self.out = None
        self.inputs = z #- np.max(z, axis=1, keepdims=True)
        pos_exp = np.exp(self.inputs)
        neg_exp =  np.exp(-1. * (self.inputs))
        self.output = (pos_exp - neg_exp) / (pos_exp + neg_exp)
        return self.output

    def backward(self, dvalues):
        """
        Hyperbolic Tangent Backward.

        z --> (Tanh) --> self.out
        dz <-- (dTanh) <-- d_prev(dL/d self.out)

        [Inputs]
            d_prev : Gradients flow from upper layer.
            reg_lambda: L2 regularization weight. (Not used in activation function)

        [Outputs]
            dz : Gradients w.r.t. Tanh input z .
            In other words, the derivative of tanh should be reflected on d_prev.
        """
        dz = None
        dz = self.dinputs = (1.0 - self.output**2) * dvalues
        return dz
 
class Softmax:
    def __call__(self,inputs):
        return self.forward(inputs)
    def forward(self,inputs, training):
        self.inputs = inputs
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
            single_output = single_output.reshape(-1,1)
            # Calculate Jacobian matrix of the output
                                #S_i_j * sigma_j_k      -     S_i_j * S_i_k 
            jacobian_matrix = np.diagflat(single_output) - np.dot(single_output, single_output.T)
            # Calculate sample-wise gradient
            # and add it to the array of sample gradients
            self.dinputs[index] = np.dot(jacobian_matrix, single_dvalues)
    def predictions(self, outputs):
        return np.argmax(outputs, axis=1)
class Sigmoid:
    def __call__(self,inputs):
        return self.forward(inputs)
    def forward(self,inputs, training):
        inputs = -inputs
        exp_values = np.exp(inputs - np.max(inputs, axis = 1, keepdims = True))
        self.output = 1 / (1 + exp_values)
        return self.output
    def backward(self, dvalues):
        return dvalues * self.output * (1 - self.output)
    # Calculate predictions for outputs
    def predictions(self, outputs):
        return (outputs > 0.5) * 1
class Linear:
    #Forward pass 
    def forward(self, inputs, training):
        #just remember values
        self.inputs = inputs
        self.output = output
    #Backward pass
    def backward(self, dvalues):
        #derivatice is 1, 1 * dvalues = dvalues
        self.dinputs = dvalues.copy()
    # Calculate predictions for outputs
    def predictions(self, outputs):
        return outputs
