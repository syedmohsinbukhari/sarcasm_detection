# -*- coding: utf-8 -*-
"""
Created on Mon May  7 12:00:00 2018

@author: elcid
"""

""" Setup logging and environment """
# simulate that sarcasmdetection is installed as a python package
import context

"""--------------------------------------------------"""
import torch
import torchtext

import numpy as np
import logging
import pickle
import matplotlib.pyplot as plt

import sarcasmdetection as sd

"""--------------------------------------------------"""
sd.utils.setup_logging('logs/plot_pca.log')
logging.info("Running script plot_pca.py")

"""--------------------------------------------------"""
test_embeddings = pickle.load(open('data/models/test_embeddings.pkl', 'rb'))
test_labels = pickle.load(open('data/models/test_labels.pkl', 'rb'))

test_labels = np.array(test_labels)

def PCA(data, k=2):
    # preprocess the data
    X = torch.from_numpy(data)
    X_mean = torch.mean(X,0)
    X = X - X_mean.expand_as(X)

    # svd
    U,S,V = torch.svd(torch.t(X))
    return torch.mm(X,U[:,:k])

def reject_outliers(data, labels, m=2):
    indices_x = abs(data[:,0] - np.mean(data[:,0])) < m * np.std(data[:,0])
    indices_y = abs(data[:,1] - np.mean(data[:,1])) < m * np.std(data[:,1])
    indices = indices_x + indices_y
    indices = abs(data[:,0]) < m * np.std(data[:,0])
    return data[indices], labels[indices]

X_PCA = PCA(test_embeddings).numpy()
X_PCA, test_labels = reject_outliers(X_PCA, test_labels, m=2)

plt.figure()

S_X = X_PCA[[i for i in range(len(test_labels)) if test_labels[i]==1]]
N_X = X_PCA[[i for i in range(len(test_labels)) if test_labels[i]==0]]

# print(S_X)
plt.scatter(S_X[:, 0], S_X[:, 1], label='sarcastic', alpha=0.5)
plt.scatter(N_X[:, 0], N_X[:, 1], label='non-sarcastic', alpha=0.5)

plt.legend()
# plt.title('PCA of IRIS dataset')
plt.show()

"""--------------------------------------------------"""
logging.info("Finished script plot_pca.py")
