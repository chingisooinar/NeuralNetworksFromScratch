#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  6 07:34:11 2021

@author: nuvilabs
"""
import numpy as np
import matplotlib.pyplot as plt
import nnfs
from nnfs.datasets import spiral_data
nnfs.init()
class Layer_Dense:
    # Layer initialization
    def __init__(self, n_inputs, n_neurons):
        """
        We’re going to multiply this
        Gaussian distribution for the weights by 0.01 to generate numbers that are a couple of magnitudes
        smaller. Otherwise, the model will take more time to fit the data during the training process
        as starting values will be disproportionately large compared to the updates being made during
        training. The idea here is to start a model with non-zero values small enough that they won’t affect
        training. This way, we have a bunch of values to begin working with, but hopefully none too large
        or as zeros. You can experiment with values other than 0.01 if you like.
        """
        self.weights = 0.01 * np.random.randn(n_inputs,n_neurons)
        self.biases = np.zeros((1, n_neurons))
    # Forward pass
    def forward(self, inputs):
        # Calculate output values from inputs, weights and biases
        self.output = np.dot(inputs, self.weights) + self.biases

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