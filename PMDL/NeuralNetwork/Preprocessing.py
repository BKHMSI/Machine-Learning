import numpy as np
from scipy.ndimage.interpolation import shift, rotate, zoom

class Normalize(object):
    def __init__(self):
        self.img_mean = self.img_std = None
    
    def norm_image(self, X):
        if self.img_mean is None:
            self.img_mean = np.mean(X, axis=0)
            self.img_std = np.std(X, axis=0)
        return (X - self.img_mean) / self.img_std 
    
class ImageProcessing(object):
    def __init__(self):
        pass

    def to_cifar10_2d(self, X):
        return X.reshape(-1,32,32,3)

    def to_cifar10_1d(self, X):
        return X.reshape(X.shape[0],-1)

    def augment_image(self, X, seed):
        augment = {
            0: self.shift,
            1: self.flip_h,
            2: self.flip_v,
            3: self.zoom 
        }
        return augment[seed](X)
    
    def rotate(self, X, deg):
        X_rotate = self.to_cifar10_2d(X)
        return self.to_cifar10_1d(X_rotate)
    
    def shift(self, X):
        X_shift = self.to_cifar10_2d(X)
        shift = np.random.randint(X_train.shape[2])
        X_shift = np.roll(X_train,shift,axis=2)
        return self.to_cifar10_1d(X)

    def flip_h(self, X):
        X_flip = self.to_cifar10_2d(X)
        X_flip = np.fliplr(X_flip)
        return self.to_cifar10_1d(X_flip)
    
    def flip_v(self, X):
        X_flip = self.to_cifar10_2d(X)
        X_flip = np.flipud(X_flip)
        return self.to_cifar10_1d(X_flip)
