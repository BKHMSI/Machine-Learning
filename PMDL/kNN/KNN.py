# -*- coding: utf-8 -*-
"""
Created on Tue Jan 17 13:41:10 2017

"""
import numpy as np
from collections import Counter
from scipy.spatial import distance

class KNN(object):
    
    def __init__(self):
        pass

    def train(self, X, y):
        """ X is N x D where each row is an example. Y is 1-dimension of size N """
        # the nearest neighbor classifier simply remembers all the training data
        self.Xtr = X
        self.ytr = y

    def predict(self, X, l='L2', k = 5):
        """ X is N x D where each row is an example we wish to predict label for """
        num_test = X.shape[0]
        # lets make sure that the output type matches the input type

        Ypred = np.zeros(num_test, dtype = self.ytr.dtype)
        if l == 'L2':
            a_sq = np.sum(np.square(self.Xtr), axis = 1)
            dist = np.sqrt(a_sq[:,np.newaxis] + np.sum(np.square(X), axis = 1) - 2*(self.Xtr.dot(X.T)))
            dist = np.transpose(dist)
            #dist = distance.cdist(X,self.Xtr,'euclidean')
        else:
            dist = distance.cdist(X,self.Xtr,'cityblock')
       
        min_indecies = np.argpartition(dist, k, axis=1)[:, :k]
   
        # predict the label of the k nearest neighbors using majority voting
        for i in xrange(min_indecies.shape[0]):
            Ypred[i] = Counter(self.ytr[min_indecies[i,:]]).most_common(1)[0][0]
        return Ypred
