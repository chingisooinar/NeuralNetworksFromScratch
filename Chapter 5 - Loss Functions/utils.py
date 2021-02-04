#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 23 07:40:29 2021

@author: nuvilabs
"""
import numpy as np
class AccuracyMeter:
    def __call__(self, preds, true):
        y_hat = np.argmax(preds, axis = 1)
        #for one-hot encoded labels
        if len(true.shape) == 2:
            true = np.argmax(true, axis = 1)
        return np.mean(y_hat == true)
