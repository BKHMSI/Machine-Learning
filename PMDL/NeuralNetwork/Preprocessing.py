import numpy as np
from scipy.ndimage.interpolation import shift


class Normalize(object):
    def __init__(self):
        self.img_mean = self.img_std = None
    
    def norm_image(self, X):
        if self.img_mean is None:
            self.img_mean = np.mean(X, axis=0)
            self.img_std = np.std(X, axis=0)
        return (X - self.img_mean) / self.img_std 

class PCA(object):
    def __init__(self, k):
        self.U = None
        self.k = k
    
    def process(self, X):
        if self.U is None:
            cov = (X.T.dot(X)) / X.shape[0]
            self.U, _, _ = np.linalg.svd(cov)
        return X.dot(self.U)[:,:self.k]
    
class ImageProcessing(object):
    def __init__(self):
        pass

    def to_cifar10_2d(self, X):
        return X.reshape(-1,32,32,3)

    def to_cifar10_1d(self, X):
        return X.reshape(X.shape[0],-1)

    def augment_image(self, X, seed):
        #return self.original(X)
        augment = {
            0: self.original,
            1: self.original,
            2: self.flip_h,
            3: self.flip_h,
            4: self.shift_img
        }
        return augment[seed](X)
    
    def original(self, X):
        return X
    
    def rotate(self, X):
        X_rotate = self.to_cifar10_2d(X)
        deg = np.random.randint(10)
        print ("Rotating by ", deg, " degrees")
        for i, x in enumerate(X_rotate):
            X_rotate[i] = scipy.misc.imrotate(x,deg)
        return self.to_cifar10_1d(X_rotate)
    
    def shift_img(self, X):
        X_shift = self.to_cifar10_2d(X)
        s = np.random.randint(1,2)
        print ("Shifting by ", s, " pixels")
        X_shift = np.roll(X_shift,s,axis=2)
        #X_shift = shift(X_shift,[0,s*v,s*h,0])
        return self.to_cifar10_1d(X_shift)

    def flip_h(self, X):
        print( "Fliping Horizontally")
        X_flip = self.to_cifar10_2d(X)
        for i, x in enumerate(X_flip):
            X_flip[i] = np.fliplr(x)
        return self.to_cifar10_1d(X_flip)
    
    def flip_v(self, X):
        print ("Fliping Vertically")
        X_flip = self.to_cifar10_2d(X)
        for i, x in enumerate(X_flip):
            X_flip[i] = np.flipud(x)
        return self.to_cifar10_1d(X_flip)
