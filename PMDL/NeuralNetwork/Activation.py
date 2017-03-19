import numpy as np
from scipy.stats import logistic

class Activation(object):
    def __init__(self):
        pass

    def set_function(self, activation):
        self.activation = activation

    def softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()

    def sigmoid(self, x):
        return logistic.cdf(x)
        #return 1.0 / (1.0 + np.exp(-x))

    def sigmoid_gradient(self, x):
        gg = self.sigmoid(x)
        return gg  * (1 - gg)

    def relu(self, x):
        return np.maximum(0,x)
    
    def relu_gradient(self, x):
        return (x > 0)

    def leaky_relu(self, x):
        return np.maximum(0.1*x, x)

    def elu(self, x):
        return np.maximum(np.exp(x) - 1, x)

    def tanh(self, x):
        return np.tanh(x)
    
    def tanh_gradient(self, x):
        return 1 - np.tanh(x)**2

    def gradient(self, x, activation):
        if(activation == 'relu'):
            return self.relu_gradient(x)
        elif(activation == 'sigmoid'):
             return self.sigmoid_gradient(x)
        elif(activation == 'tanh'):
             return self.tanh_gradient(x)
        elif(activation == 'softmax'):
             return 1

    def fire(self, x, activation):
        if(activation == 'relu'):
            return self.relu(x)
        elif(activation == 'sigmoid'):
             return self.sigmoid(x)
        elif(activation == 'tanh'):
             return self.tanh(x)
        elif(activation == 'softmax'):
             return self.softmax(x)
