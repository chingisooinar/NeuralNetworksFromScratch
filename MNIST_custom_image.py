#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 13 18:42:35 2021

@author: nuvilabs
"""
from zipfile import ZipFile
import os
import urllib
import urllib.request
import numpy as np
import cv2
from MNIST_utils import create_data_mnist
from model import Model
from layers.Dense import Layer_Dense
from layers.dropout import Dropout
from activations.activations import *
from losses.losses import *
from optimizers.optimizers import *
from utils.utils import *
LABELS = ['Tshirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
#load custom image
img_data = cv2.imread('tshirt.png', cv2.IMREAD_GRAYSCALE)
#visualize an image
import matplotlib.pyplot as plt
plt.imshow(img_data,cmap='gray') 
plt.show()
#resize 
img_data = cv2.resize(img_data,(28,28))
plt.imshow(img_data,cmap='gray') 
plt.show()
#invert image colors
img_data = 255 - img_data
#reshape and normalize
img_data = (img_data.reshape(1,-1).astype(np.float32) - 127.5) / 127.5
# Instantiate the model
model = model = Model.load('fashion_mnist.model')

y_hat = model.predict(img_data)

print(f'prediction: {LABELS[y_hat[0]]}')