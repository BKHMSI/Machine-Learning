import numpy as np
import matplotlib.pyplot as plt
from NeuralNetwork import ANN

np.random.seed(1337)  # for reproducibility\n",

num_train = 10000
num_test = 1000

Xtr = np.random.randint(2, size=(num_train, 2))
ytr =  Xtr[:,0] ^ Xtr[:,1]

X_test = np.random.randint(2, size=(num_test, 2))
y_test = X_test[:,0] ^ X_test[:,1]

classifier = ANN()
classifier.add_input(Xtr.shape[1])
classifier.add_layer(2, 'relu')
classifier.add_layer(2, 'sigmoid')
classifier.summary()

classifier.train(Xtr, ytr, reg = 1e-6, epochs = 50, batch_size = 100, alpha = 1e-5, validation_split = 0.1)

y_test_pred = classifier.predict(X_test)
num_correct = np.sum(y_test_pred == y_test)
print "Accuracy: ", float(num_correct) / num_test


#train(self, X, y, reg = 0, epochs = 10, batch_size = 32):
