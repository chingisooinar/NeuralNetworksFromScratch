#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 23 07:03:46 2021

@author: nuvilabs
"""
from activations import Softmax
import numpy as np
class Loss:
    def regularization_loss(self,layer):
        regularization_loss = 0
        #weights
        if layer.weight_regularizer_l1:
            regularization_loss += layer.weight_regularizer_l1 * np.sum(np.abs(layer.weights))
        if layer.weight_regularizer_l2:
            regularization_loss += layer.weight_regularizer_l2 * np.sum(layer.weights**2)
        #biases
        if layer.bias_regularizer_l1:
            regularization_loss += layer.bias_regularizer_l1 * np.sum(np.abs(layer.biases))
        if layer.bias_regularizer_l2:
            regularization_loss += layer.bias_regularizer_l2 * np.sum(layer.biases**2)
            
        return regularization_loss
            
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
        
class BinaryCrossentropy(Loss):
    #forward pass
    def forward(self, y_pred, y_true):
        self.y_true = y_true
        #Clip both sides to not drag mean towards any value
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)
        #Calculate sample-wise loss
        sample_losses = -(y_true * np.log(y_pred_clipped) + (1 - y_true) * np.log(1 - y_pred_clipped))
        sample_losses = np.mean(sample_losses, axis=-1)
        
        #return losses
        return sample_losses
    #backward pass
    def backward(self, dvalues):
        y_true = self.y_true
        samples = len(dvalues)
        #We'll use the first sample to count them
        outputs = len(dvalues[0])
        
        #Clip both sides to not drag mean towards any value
        clipped_dvalues = np.clip(dvalues, 1e-7, 1 - 1e-7)
        
        self.dinputs = -(y_true / clipped_dvalues - (1 - y_true) / (1 - clipped_dvalues)) / outputs
        
        #normalize gradient
        self.dinputs = self.dinputs / samples

class MSE(Loss):
    # forward pass
    def forward(self, y_pred, y_true):
        #calculate loss
        samples_losses = np.mean((y_true - y_pred)**2, axis=-1)
        #return losses
        return sample_losses
    def backward(self, dvalues, y_true):
        #number of samples
        samples = len(dvalues)
        #number of outputs in every sample
        outputs = len(dvalues[0])
        
        #gradient on values
        self.dinputs = -2 * (y_true - dvalues) / outputs
        #normalize gradient
        self.dinputs = self.dinputs / samples

class MAE(Loss):
    #forward pass
    def forward(self, y_pred, y_true):
        #calculate loss
        sample_losses = np.mean(np.abs(y_true - y_pred), axis=-1)
        # return losses
        return sample_losses
    #backward pass
    def backward(self, dvalues, y_true):
        #number of samples
        samples = len(dvalues)
        #number of outputs in every sample
        outputs = len(dvalues[0])
        
        #gradient on values
        self.dinputs = np.sign(y_true - dvalues) / outputs
        #normalize gradient
        self.dinputs = self.dinputs / samples