#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  5 07:16:54 2021

@author: nuvilabs
"""

import numpy as np
import matplotlib.pyplot as plt
import nnfs
from nnfs.datasets import spiral_data
class Loss:
    def calculate(self, output, y):
        #Calculate sample losses
        sample_losses = self.forward(output, y)
        #Calculate mean loss
        data_loss = np.mean(sample_losses)
        
        return data_loss
    def __call__(self, output, y):
        return self.calculate(output,y)
    
class CatCrossentropy(Loss):
    def forward(self, y_pred, y_true):
        self.y_true = y_true
        #Number of samples in a batch
        samples = len(y_pred)
        
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)
        #for numerical labels
        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(samples),y_true]
        #for one-hot encoded labels
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(
            y_pred_clipped * y_true,
            axis=1
            )
        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods
    def backward(self,dvalues):
        #number of samples
        samples = len(dvalues)
        labels = len(dvalues[0])
        # If labels are sparse, turn them into one-hot vector
        if len(self.y_true.shape) == 1:
            y_true = np.eye(labels)[y_true]
        # Calculate gradient
        self.dinputs = -self.y_true / dvalues
        # Normalize gradient
        self.dinputs = self.dinputs / samples

class Activation_Softmax_Loss_CategoricalCrossentropy():
    def __init__(self):
        self.activation = Softmax()
        self.loss = CatCrossentropy()
        
    #forward pass
    def forward(self, inputs, y_true):
        #output layer's activation function
        self.activation.forward(inputs)
        #set the output
        self.output = self.activation.output
        
        return self.loss.calculate(self.output, y_true)
    
    #Backward pass
    def backward(self, dvalues, y_true):
        #number of samples
        samples = len(dvalues)
        #if labels are one-hot encoded, turn them into discrete values
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)
        #copy so we can safely modify
        self.dinputs = dvalues.copy()
        #yhat_i_k - y_i_k
        self.dinputs[range(samples),y_true] -= 1
        #normalize gradient
        self.dinputs /= samples
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
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases
    def backward(self, dvalues):
        #davlues.shape = (batch_size, neurons)
        self.dweights = np.dot(self.inputs.T,dvalues)  #=> (n_inputs,n_neurons)
        self.dbiases = np.sum(dvalues,axis = 0,keepdims=True) #=>(1,n_neurons)
        self.dinputs = np.dot(dvalues,self.weights.T) #=>(batch_size,n)
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
class ReLu:
    def __call__(self,inputs):
        return self.forward(inputs)
    def forward(self,inputs):
        self.inputs = inputs
        self.output =  np.maximum(0,inputs)
    def backward(self,dvalues):
        self.dinputs = np.copy(dvalues)
        self.dinputs[self.inputs <= 0] = 0
    
class Softmax:
    def __call__(self,inputs):
        return self.forward(inputs)
    def forward(self,inputs):
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
nnfs.init()
X, y = spiral_data(samples=100, classes=3)
plt.scatter(X[:, 0], X[:, 1])
plt.show()
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='brg')
plt.show()
dense1 = Layer_Dense(2, 64)
# Create ReLU activation (to be used with Dense layer):
activation1 = ReLu()
# Create second Dense layer with 64 input features (as we take output
# of previous layer here) and 3 output values (output values)
dense2 = Layer_Dense(64, 3)
# Create Softmax classifier's combined loss and activation
loss_activation = Activation_Softmax_Loss_CategoricalCrossentropy()
# Create optimizer
optimizer = SGD(decay=1e-3, momentum=0.9)
# Train in loop
for epoch in range(10001):
    # Perform a forward pass of our training data through this layer
    dense1.forward(X)
    # Perform a forward pass through activation function
    # takes the output of first dense layer here
    activation1.forward(dense1.output)
    # Perform a forward pass through second Dense layer
    # takes outputs of activation function of first layer as inputs
    dense2.forward(activation1.output)
    # Perform a forward pass through the activation/loss function
    # takes the output of second dense layer here and returns loss
    loss = loss_activation.forward(dense2.output, y)
    # Calculate accuracy from output of activation2 and targets
    # calculate values along first axis
    predictions = np.argmax(loss_activation.output, axis=1)
    accuracy = np.mean(predictions==y)
    
    
    
    
    if not epoch % 100:
        print(f'epoch: {epoch}, ' +
        f'acc: {accuracy:.3f}, ' +
        f'loss: {loss:.3f}, ' +
        f'lr: {optimizer.current_lr}')
    # Backward pass
    loss_activation.backward(loss_activation.output, y)
    dense2.backward(loss_activation.dinputs)
    activation1.backward(dense2.dinputs)
    dense1.backward(activation1.dinputs)
    # Update weights and biases
    optimizer.pre_update_params()
    optimizer.update_params(dense1)
    optimizer.update_params(dense2)
    optimizer.post_update_params()