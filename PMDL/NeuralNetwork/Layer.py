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

    def update_weights(self, alpha, reg, m):
       self.regularize(reg, m)
       self.W  = self.W - ((alpha)*(self.dw))
       self.bias = self.bias - ((alpha)*(self.db))

    def regularize(self, reg, m):
        self.dw /= m
        self.db /= m
        self.dw += reg*self.W