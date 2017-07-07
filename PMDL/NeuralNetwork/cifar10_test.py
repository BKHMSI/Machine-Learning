import numpy as np
import matplotlib.pyplot as plt
from data_utils import load_CIFAR10
from NeuralNetwork import ANN 
from Preprocessing import Normalize, PCA
from keras.datasets import cifar10
#from keras.utils import np_utils

np.random.seed(1337)

classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
num_classes = len(classes)

def evaluate(y_test_pred, y_test):
    # Compute and print the fraction of correctly predicted examples

    y_test = y_test.reshape(y_test.shape[0])
    print ("Y_Pred: ", y_test.shape, "Y_Test", y_test.shape)
    num_correct = np.sum(y_test_pred == y_test)
    histogram = np.zeros(num_classes)
    num_test = y_test.shape[0]

    accuracy = float(num_correct) / num_test
    print ('Got %d / %d correct => accuracy: %f' % (num_correct, num_test, accuracy))

    for i in range(0,num_classes):
        count = np.sum(y_test==i)
        if(count != 0): 
            histogram[i-1] = (float(np.sum(y_test_pred[np.where(y_test==i)] == i)) / count) * 100

    index = np.arange(num_classes)
    bar_width = 0.5
    plt.bar(index,histogram,bar_width)
    plt.xlabel("Classes")
    plt.ylabel("% of Correctly Classified")
    plt.xticks(index+bar_width/2,classes)
    plt.show() 

# Load the raw CIFAR-10 data.
print ("Loading Data...")
cifar10_dir = '../data/cifar-10-batches-py'
X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)
#(X_train, y_train), (X_test, y_test) = cifar10.load_data()

# Mask images
num_training = 50000
mask = range(num_training)   
X_train = X_train[mask]
y_train = y_train[mask] 

num_test = 10000
mask = range(num_test)
X_test = X_test[mask]
y_test = y_test[mask]

# One-Hot Encoding
Y_train = np.eye(num_classes)[y_train]
#Y_train = np_utils.to_categorical(y_train, num_classes)

# Reshape the image data into rows
X_train = np.reshape(X_train, (X_train.shape[0], -1))
X_test = np.reshape(X_test, (X_test.shape[0], -1))

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

normalize = Normalize()
X_train = normalize.norm_image(X_train)
X_test = normalize.norm_image(X_test)

# PCA
pca = PCA(200)
X_pca = pca.process(X_train)
X_test = pca.process(X_test)


# Create Classifier
classifier = ANN()
classifier.add_input(X_pca.shape[1])
classifier.add_layer(750, 'relu')
classifier.add_layer(500, 'relu')
classifier.add_layer(num_classes, 'sigmoid')
classifier.summary()

epochs = 2

# Train classifier
classifier.train(X_train, Y_train, 
                reg = 1e-2, epochs = epochs, batch_size = 120, lr = 1e-2, 
                update_method = 'stochastic', validation_split = 0.1, pca = pca)

# Evaluate Testing Data 
y_pred_test = classifier.predict(X_test)
np.savetxt('Results/Y_Pred.txt',y_pred_test)
evaluate(y_pred_test, y_test)
classifier.show_plot(epochs)