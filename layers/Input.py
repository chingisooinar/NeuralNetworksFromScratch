#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 14 11:21:34 2021

@author: nuvilabs
"""

# Input "layer"
class Layer_Input:
# Forward pass
    def forward(self, inputs, training):
        self.output = inputs