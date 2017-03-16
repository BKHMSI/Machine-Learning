import numpy as np
from Activation import Activation

class Layer(object):
    def __init__(self, uid, units, W, bias, function):
        self.id = uid
        self.units = units 
        self.W = W
        self.bias = bias
        self.activation = function
        self.ac = Activation()
        self.m = self.v = 0
        self.mb = self.vb = 0

    def get_weights(self):
        return self.W
    
    def set_weights(self, W):
        self.W = W

    def get_bias(self):
        return self.bias

    def get_size(self):
        return self.units

    def get_func(self):
        return self.activation

    def get_gradients(self):
        return self.dw, self.db

    def set_gradients(self, dw, db):
        self.dw = dw 
        self.db = db.sum(axis=1).reshape(db.shape[0],1)

    def get_outputs(self):
        return self.a

    def set_outputs(self, a):
        self.a = a

    def set_inputs(self, z):
        self.z = z
    
    def get_inputs(self):
        return self.z
    
    def fire(self, z):
        return self.ac.fire(z, self.activation)

    def activation_gradient(self, z):
        return self.ac.gradient(z, self.activation)

    def update_weights(self, lr, reg, m, beta1 = 0.9, beta2 = 0.999, method = 'momentum'):
        self.regularize(reg, m)
        if(method == 'adam'):
          self.adam(beta1, beta2)
        elif(method == 'momentum'):
          self.momentum(0.7, lr)
        elif(method == 'stochastic'):
          self.stochastic(lr)
        
    def regularize(self, reg, m):
        self.dw /= m
        self.db /= m
        self.dw += reg*self.W

    def stochastic(self, alpha):
        self.W  = self.W - ((alpha)*(self.dw))
        self.bias = self.bias - ((alpha)*(self.db))

    def adam(self, beta1, beta2):
        eps = 1e-8
        self.m = beta1*self.m + (1-beta1)*self.dw
        self.v = beta2*self.v + (1-beta2)*(self.dw**2)
        self.mb = beta1*self.mb + (1-beta1)*self.db
        self.vb = beta2*self.vb + (1-beta2)*(self.db**2)
        self.W += - ((alpha * self.m) / (np.sqrt(self.v)+eps))
        self.bias += - ((alpha * self.mb) / (np.sqrt(self.vb)+eps))

    def momentum(self, mue, lr):
        self.v = mue * self.v - lr*self.dw
        self.vb = mue * self.vb - lr*self.db
        self.W += self.v
        self.bias += self.vb