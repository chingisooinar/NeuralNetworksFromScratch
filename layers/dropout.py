#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 10 08:03:36 2021

@author: nuvilabs
"""
"""
If we don’t use dropout during prediction, all neurons will output their
values, and this state won’t match the state seen during training, since the sums will be statistically
about twice as big. To handle this, during prediction, we might multiply all of the outputs by
the dropout fraction, but that’d add another step for the forward pass, and there is a better way
to achieve this. Instead, we want to scale the data back up after a dropout, during the training
phase, to mimic the mean of the sum when all of the neurons output their values.
"""
class Dropout:
    def __init__(self, rate):
        #Store rate, we invert it as for example for fropout
        # of 0.1 we need success rate of 0.9
        self.rate = 1 - rate
    def forward(self, inputs, training):
        # Save input values
        self.inputs = inputs
        if not training:
            self.output = inputs.copy()
            return
        # Generate and save scaled mask
        self.binary_mask = np.random.binomial(1, self.rate, size=inputs.shape) / self.rate
        # Apply mask to output values
        self.output = inputs * self.binary_mask
    def backward(self, dvalues):
        # Gradient on values
        self.dinputs = dvalues * self.binary_mask