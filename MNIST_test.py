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
#First, we will retrieve the data from the nnfs.io site. Let’s define the URL 
#of the dataset, a filename to save it locally to and the folder for the extracted images:
URL = 'https://nnfs.io/datasets/fashion_mnist_images.zip'
FILE = 'fashion_mnist_images.zip'
FOLDER = 'fashion_mnist_images'

if not os.path.isfile(FILE):
    print(f'Downloading {URL} and saving as {FILE}...')
    urllib.request.urlretrieve(URL, FILE)
    print('Unzipping images...')
    with ZipFile(FILE) as zip_images:
        zip_images.extractall(FOLDER)
    print('Done!')
#get labels
labels = os.listdir('fashion_mnist_images/train')
print(labels)

files = os.listdir('fashion_mnist_images/train/0')
print(files[:10])
print(len(files))

# Create dataset
X, y, X_test, y_test = create_data_mnist('fashion_mnist_images')

keys = np.array(range(X.shape[0]))
np.random.shuffle(keys)
X = X[keys]
y = y[keys]
# Scale features
X = (X.astype(np.float32) - 127.5) / 127.5
X_test = (X_test.astype(np.float32) - 127.5) / 127.5

# Reshape to vectors
#i.e. flatten images
X = X.reshape(X.shape[0], -1)
X_test = X_test.reshape(X_test.shape[0], -1)



import matplotlib.pyplot as plt
plt.imshow((X[8].reshape(28, 28))) # Reshape as image is a vector already
plt.show()

# Instantiate the model
model = model = Model.load('fashion_mnist.model')

model.evaluate(X_test, y_test, batch_size=128)
y_hat = model.predict(X_test, batch_size=128)
from sklearn.metrics import f1_score
print(f'f1 score:{f1_score(y_test, y_hat,average="macro")}')