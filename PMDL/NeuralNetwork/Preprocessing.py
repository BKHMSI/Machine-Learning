import numpy as np

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