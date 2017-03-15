import numpy as np
from collections import Counter
from numpy.linalg import inv

class LLS(object):
    def __init__(self):
        pass

    def train(self, X, y):
        """ X is N x D where each row is an example. Y is 1-dimension of size N """
        num_classes = np.unique(y).shape[0]
        self.W = np.zeros((num_classes, X.shape[1]))

        for i in range(0,num_classes):
            self.W[i] = inv((X.T).dot(X)).dot(X.T).dot((y==i)*1)

    def predict(self, x):
        Y = self.W.dot(x.T)
        return np.argmax(Y, axis=0)
