#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  6 06:31:42 2021

@author: nuvilabs
"""

#toy input
inputs = [1.0, 2.0, 3.0, 2.5]
#toy weights
weights1 = [0.2, 0.8, -0.5, 1.0]
weights2 = [0.5, -0.91, 0.26, -0.5]
weights3 = [-0.26, -0.27, 0.17, 0.87]
weights = [weights1, weights2, weights3]
#random bias
bias1 = 2.0
bias2 = 3.0
bias3 = 0.5
biases = [bias1, bias2, bias3]
"""
This neuron sums each input multiplied by that input’s weight, then adds the bias. All the neuron
does is take the fractions of inputs, where these fractions (weights) are the adjustable parameters,
and adds another adjustable parameter — the bias — then outputs the result.
"""
# Output of current layer
layer_outputs = []
# For each neuron
for neuron_weights, neuron_bias in zip(weights, biases):
    # Zeroed output of given neuron
    neuron_output = 0
    # For each input and weight to the neuron
    for n_input, weight in zip(inputs, neuron_weights):
        # Multiply this input by associated weight
        # and add to the neuron's output variable
        neuron_output += n_input*weight
    # Add bias
    neuron_output += neuron_bias
    # Put neuron's result to the layer's output list
    layer_outputs.append(neuron_output)
print(layer_outputs) # >>> (4.8, 1.21, 2.385)
#A Layer of Neurons with NumPy
"""
Previously, we had calculated outputs of each neuron by performing a dot product and adding a
bias, one by one. Now we have changed the order of those operations — we’re performing dot
product first as one operation on all neurons and inputs, and then we are adding a bias in the next
operation. When we add two vectors using NumPy, each i-th element is added together, resulting
in a new vector of the same size. This is both a simplification and an optimization, giving us
simpler and faster code.
"""
import numpy as np
outputs = np.dot(weights, inputs) + biases
print(outputs) # >>> (4.8, 1.21, 2.385)
#A Batch of Data
"""
Often, neural networks expect to take in many samples at a time for two reasons. One reason
is that it’s faster to train in batches in parallel processing, and the other reason is that batches
help with generalization during training. If you fit (perform a step of a training process) on one
sample at a time, you’re highly likely to keep fitting to that individual sample, rather than slowly
producing general tweaks to weights and biases that fit the entire dataset. Fitting or training in
batches gives you a higher chance of making more meaningful changes to weights and biases.
"""
inputs = [[1, 2, 3, 2.5], 
          [2, 5, -1, 2], 
          [-1.5, 2.7, 3.3, -0.8]]
outputs = np.dot(inputs, np.asarray(weights).T) + biases
print(outputs)