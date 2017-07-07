
# coding: utf-8

# In[1]:

import os
import cv2 as cv
import numpy as np
#import matplotlib.pyplot as plt
import scipy.misc

from keras.applications import Xception
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D, Input, Dropout
from keras import backend as K
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, TensorBoard, ReduceLROnPlateau

epochs = 20
train_size = 90000
val_size = 10000
batch_size = 16
batch_val_size = 16
input_dim = (299, 299) # inception_v3 input dimensions 

# Data Augmentation
train_datagen = ImageDataGenerator(rescale=1./255,
                            zoom_range = 0.2,
                            shear_range = 0.2,
                            #featurewise_center=True,  # set input mean to 0 over the dataset
                            samplewise_center=False,  # set each sample mean to 0
                            #featurewise_std_normalization=True,  # divide inputs by std of the dataset
                            samplewise_std_normalization=False,  # divide each input by its std
                            rotation_range= 10,  # randomly rotate images in the range (degrees, 0 to 180)
                            width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
                            height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
                            horizontal_flip=True)  # randomly flip images

test_datagen = ImageDataGenerator(rescale=1./255)


# create the base pre-trained model
base_model = Xception(weights='imagenet', include_top=False, input_tensor=Input(shape=(299, 299, 3)))


# In[7]:

# add a global spatial average pooling layer
x = base_model.output
x = GlobalAveragePooling2D()(x)
# and a logistic layer -- let's say we have 200 classes
predictions = Dense(200, activation='softmax')(x)


# In[8]:

# this is the model we will train
model = Model(input=base_model.input, output=predictions)



# In[12]:

model.load_weights("weights20_01_0.80.hdf5")


# In[ ]:
from keras.preprocessing import image as image_utils
#from xception import preprocess_input
import numpy as np


def classify(results_file, test_files, classes_file, model, img_size):
    fd = open(results_file, 'w')
    fd2 = open(classes_file, 'r')

    classes = []
    for i in range(200):
        my_str = fd2.readline()
        classes.append(my_str)

    new_classes = sorted(classes, key=lambda item: (int(item.partition(' ')[0])
                                                    if item[0].isdigit() else float('inf'), item))
    fd2.close()

    print("[INFO] classifying image...")

    for i in range(10000):
        file_name = 'test_%d.JPEG' % (i,)
        image = image_utils.load_img(test_files + file_name, target_size=(img_size, img_size))
        image = image_utils.img_to_array(image)

        image = np.expand_dims(image, axis=0)
        #image = preprocess_input(image)
        image *= 1./255;

        predictions = model.predict(image)
        class_index = np.argmax(predictions, axis=-1)[0]
        class_id = new_classes[class_index]

        line = file_name + ',' + class_id
        if '\n' not in class_id:
            line += '\n'

        fd.write(line)
        if i%500 == 0:
            print (i)

    fd.close()

classify('MicroImageNet/predictions',
'MicroImageNet/test_images/',
'MicroImageNet/wnids.txt',model,299)
