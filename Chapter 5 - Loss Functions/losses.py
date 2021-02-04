#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 23 07:03:46 2021

@author: nuvilabs
"""
import numpy as np
class Loss:
    def calculate(self, output, y):
        #Calculate sample losses
        sample_losses = self.forward(output, y)
        #Calculate mean loss
        data_loss = np.mean(sample_losses)
        
        return data_loss
    def __call__(self, output, y):
        return self.calculat(output,y)
    
class CatCrossentropy(Loss):
    def forward(self, y_pred, y_true):
        #Number of samples in a batch
        samples = y_pred.shape[0]
        
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)
        #for numerical labels
        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[:,y_ture]
        #for one-hot encoded labels
        elif len(y_true.shape) == 2:
            correct_confidences = y_pred_clipped[y_true == 1]
        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods
