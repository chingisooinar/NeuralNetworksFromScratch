#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  2 08:57:00 2021

@author: nuvilabs
"""
[]
import numpy as np
import matplotlib.pyplot as plt
import nnfs
from nnfs.datasets import spiral_data
nnfs.init()
class Layer_Dense:
    # Layer initialization
    def __init__(self, n_inputs, n_neurons, weight_regularizer_l1=0, weight_regularizer_l2=0, bias_regularizer_l1=0, bias_regularizer_l2=0):
        """
        We’re going to multiply this
        Gaussian distribution for the weights by 0.01 to generate numbers that are a couple of magnitudes
        smaller. Otherwise, the model will take more time to fit the data during the training process
        as starting values will be disproportionately large compared to the updates being made during
        training. The idea here is to start a model with non-zero values small enough that they won’t affect
        training. This way, we have a bunch of values to begin working with, but hopefully none too large
        or as zeros. You can experiment with values other than 0.01 if you like.
        """
        self.weights = 0.1 * np.random.randn(n_inputs,n_neurons)
        self.biases = np.zeros((1, n_neurons))
        self.weight_regularizer_l1 = weight_regularizer_l1
        self.weight_regularizer_l2 = weight_regularizer_l2
        self.bias_regularizer_l1 = bias_regularizer_l1
        self.bias_regularizer_l2 = bias_regularizer_l2
    # Forward pass
    def forward(self, inputs):
        # Calculate output values from inputs, weights and biases
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases
    def backward(self, dvalues):
        #davlues.shape = (batch_size, neurons)
        self.dweights = np.dot(self.inputs.T,dvalues)  #=> (n_inputs,n_neurons)
        self.dbiases = np.sum(dvalues,axis = 0,keepdims=True) #=>(1,n_neurons)
        #L1 on weights
        if self.weight_regularizer_l1 > 0:
            dL1 = np.ones_like(self.weights)
            dL1[self.weights < 0] = -1
            self.dweights += self.weight_regularizer_l1 * dL1
        #L2 on weights
        if self.weight_regularizer_l2 > 0:
            self.dweights += 2 * self.bias_regularizer_l2 * self.weights
        #L1 on biases
        if self.bias_regularizer_l1 > 0:
            dL1 = np.ones_like(self.biases)
            dL1[self.biases < 0] = -1
            self.dbiases += self.bias_regularizer_l1 * dL1
        #L2 on biases
        if self.bias_regularizer_l2 > 0:
            self.dbiases = 2 * self.bias_regularizer_l2 * self.biases
        # Gradient on values
        self.dinputs = np.dot(dvalues,self.weights.T) #=>(batch_size,n)
# Create dataset        
X, y = spiral_data(samples=100, classes=3)
plt.scatter(X[:, 0], X[:, 1])
plt.show()
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='brg')
plt.show()
# Create Dense layer with 2 input features and 3 output values
dense1 = Layer_Dense(2, 3)
# Perform a forward pass of our training data through this layer
dense1.forward(X)
# Let's see output of the first few samples:
print(dense1.output[:5])