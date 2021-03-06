import numpy as np
import matplotlib.pyplot as plt
from Layer import Layer
from Preprocessing import ImageProcessing, PCA

class ANN(object):
    def __init__(self):
        pass
        
    def add_input(self, inputdim):
        self.arch = []
        self.arch.append(Layer(0, inputdim, np.zeros(0), np.zeros(0), 'no'))
        self.prev = inputdim 

    def add_layer(self, size, activation = 'relu', dropout = 1):
        w = self.init_weights(size, self.prev)
        b = self.init_weights(size, 1)
        self.arch.append(Layer(len(self.arch), size, w, b, activation, dropout))
        self.prev = size 
    
    def init_weights(self, D, H):
        return (np.random.randn(D,H) / np.sqrt(D/2))

    def backward_propagate(self, y, out):
        """Backward Propogation for one Batch"""
        num_layers = len(self.arch)-1

        # ( a(l) - y )
        delta = ( out -  y.T ) 

        # a(l-1)
        prev_out = self.arch[num_layers-1].get_outputs()

        # delta(l).a(l-1)' 
        dw = delta.dot(prev_out.T)

        # store gradients: dw and db 
        self.arch[num_layers].set_gradients(dw, delta)

        for i in range(num_layers-1,0,-1):

            next_layer = self.arch[i+1]
            layer = self.arch[i]
            prev_layer = self.arch[i-1]

            # get weights of layer i
            wi = layer.get_weights() 

            # W(i+1)'.delta(i+1)
            wd = (next_layer.get_weights().T).dot(delta)

            # (W(i+1).delta(i+1)) * f'(zi)
            delta = (wd * layer.activation_gradient(layer.get_inputs())) * layer.get_dropout()

            # delta(i).x(i-1)'
            dwi = delta.dot(prev_layer.get_outputs().T) 

            # store gradients: dw and db 
            layer.set_gradients(dwi,delta)


    def feed_forward(self, X, training = False):
        """Feed-forward for one Batch: returns output of last layer"""
        num_layers = len(self.arch)
        a = X.T # transpose to make it features x number of examples
        self.arch[0].set_outputs(a) # store values of first output
        for i, layer in enumerate(self.arch[1:num_layers]):
            # f ( W.x + b )
            layer.set_dropout()
            z = layer.get_weights().dot(a)+layer.get_bias()
            a = layer.fire(z) * layer.get_dropout(training)
            # save inputs and outputs of this layer for back propagation
            layer.set_inputs(z)
            layer.set_outputs(a)
        return a

    
    def get_weights(self):
        w = np.zeros(0)
        for layer in self.arch[1:]:
            w = np.append(w,layer.get_weights().flatten())
        return w          
    
    def get_batch(self, idx, batch_size, X):
        start = idx*batch_size
        end = (idx+1)*batch_size
        return X[start:end], self.ytr[start:end]

    def split_data(self, split_ratio, X, y):
        border = int(X.shape[0]*split_ratio)
        X_valid = X[:border]
        y_valid = y[:border]
        X_train = X[border:]
        y_train = y[border:]
        return X_train, y_train, X_valid, y_valid

    def train(self, X, y, reg = 0, epochs = 10, batch_size = 32, lr = 0.01, 
              beta1 = 0.9, beta2 = 0.999, update_method = 'momentum', 
              validation_split = 0.1, verbose = 1, pca = None):

        self.Xtr, self.ytr, Xval, yval = self.split_data(validation_split,X,y)
        self.num_classes = np.unique(y).shape[0]

        augment = ImageProcessing()

        m = self.Xtr.shape[0] # of training examples
        n = self.Xtr.shape[1] # of features
        batches = int(m / batch_size)
        val_size = Xval.shape[0]

        Xval = pca.process(Xval)

        # Store data for plotting
        self.val_loss = []
        self.val_acc = []
        self.train_loss = []
        self.train_acc = []	
	
        for i in range(epochs):

            Xtr = pca.process(self.Xtr)

            # Validation Pass
            probs = self.feed_forward(Xval)
            vloss = self.cost_function(probs.T, yval, reg, val_size)
            vacc = self.accuracy(Xval, yval)

            # Training Pass
            probs = self.feed_forward(Xtr,True)
            tloss = self.cost_function(probs.T, self.ytr, reg, m)
            tacc = self.accuracy(Xtr, self.ytr)
            if(verbose):
                print ("\nEpoch #", i+1)
                print ("val_loss: ", vloss, " | val_acc: ", vacc)
                print( "train_loss: ", tloss, " | train_acc: ",  tacc)

            self.val_loss.append(vloss)
            self.val_acc.append(vacc)
            self.train_loss.append(tloss)
            self.train_acc.append(tacc)


            option = np.random.randint(5)
            X_aug = augment.augment_image(self.Xtr,option)
            X_aug = pca.process(X_aug)

            for j in range(batches):
                xtr, ytr = self.get_batch(j, batch_size, X_aug) 
                # Feed Forward
                out = self.feed_forward(xtr,True)
                # Backward Propagation 
                self.backward_propagate(ytr, out)
                # Update Weights
                for layer in self.arch[1:]:
                    layer.update_weights(lr, reg, batch_size, method=update_method)
	
        w1 = self.arch[1].get_weights()
        w2 = self.arch[2].get_weights()
        b1 = self.arch[1].get_bias()
        b2 = self.arch[2].get_bias()

        np.savetxt('Results/Weights1.txt',w1)
        np.savetxt('Results/Weights2.txt',w2)
        np.savetxt('Results/Biases1.txt',b1)
        np.savetxt('Results/Biases2.txt',b2)
        np.savetxt('Results/ValLoss.txt',self.val_loss)
        np.savetxt('Results/TrainLoss.txt',self.train_loss)

    def predict(self, x):
        probs = self.feed_forward(x)
        return np.argmax(probs, axis=0)

    def show_plot(self, epochs):
        epoch_axis = np.arange(epochs)
        plt.plot(epoch_axis,self.val_loss,'xr-')
        plt.plot(epoch_axis,self.train_loss,'xb-')
        plt.show()

    def load_weights(self):
        b1 = np.loadtxt('Results/Biases1.txt')
        b2 = np.loadtxt('Results/Biases2.txt')
        w1 = np.loadtxt('Results/Weights1.txt')
        w2 = np.loadtxt('Results/Weights2.txt')
        print ("Weights Loaded")
        self.arch[1].set_weights(w1)
        self.arch[2].set_weights(w2)
        self.arch[1].set_bias(b1)
        self.arch[2].set_bias(b2)
   
    def accuracy(self, x, Y):
        y_pred = self.predict(x)
        y = np.argmax(Y,axis=1) 
        num_correct = np.sum(y_pred == y)
        return float(num_correct) / Y.shape[0]

    def summary(self):
       print ("\nNetwork Summary: ")
       for l, layer in enumerate(self.arch):
           print ("Layer #", l, ": Size ", layer.get_size(), "using ", layer.get_func(), " activation", " with dropout of ", layer.get_dropout(False))

    def cost_function(self, a, y, reg, m):
        R = ( reg / 2) * np.sum(self.get_weights()**2)
        L = - np.sum(np.log(self.softmax_cost(a,y)))
        return   float(L + R) / m   
    
    def softmax_cost(self, a, y):
        e_x = np.exp((a.T - np.max(a,axis=1)).T)
        return e_x[y==1] / e_x.sum(axis=1)