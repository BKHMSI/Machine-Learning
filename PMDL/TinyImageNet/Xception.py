
# coding: utf-8

# In[1]:

import os
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import scipy.misc

from keras.applications import Xception
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D, Input, Dropout
from keras import backend as K
from keras.optimizers import SGD, Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image as image_utils
from keras.callbacks import ModelCheckpoint, TensorBoard, ReduceLROnPlateau, Callback


# In[2]:

epochs = 20
train_size = 90000
val_size = 10000
batch_size = 20
batch_val_size = 32
input_dim = (256, 256) 


# In[3]:

class SGDLearningRateTracker(Callback):
    def on_epoch_end(self, epoch, logs={}):
        optimizer = self.model.optimizer
        lr = K.eval(optimizer.lr * (1. / (1. + optimizer.decay * optimizer.iterations)))
        print('\nLR: {:.6f}\n'.format(lr))

# tensorboard --logdir=.
md = ModelCheckpoint(filepath='./weights18_{epoch:02d}_{val_acc:.2f}.hdf5', monitor='val_loss', verbose=0, save_best_only=True,save_weights_only=False,mode='auto',period=1)
tb = TensorBoard(log_dir='./logs/18', histogram_freq=0, write_graph=True, write_images=False)


# In[4]:

# Data Augmentation
train_datagen = ImageDataGenerator(rescale=1./255,
                            zoom_range = 0.2,
                            shear_range = 0.2,
                            samplewise_center=False,  # set each sample mean to 0
                            samplewise_std_normalization=False,  # divide each input by its std
                            rotation_range= 10,  # randomly rotate images in the range (degrees, 0 to 180)
                            width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
                            height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
                            horizontal_flip=True)  # randomly flip images

test_datagen = ImageDataGenerator(rescale=1./255)


# In[5]:
train_generator = train_datagen.flow_from_directory(
        '/home/balkhamissi/Desktop/Assignment 3/MicroImageNet/train',
        target_size=input_dim,
        batch_size=batch_size,
        class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
        '/home/balkhamissi/Desktop/Assignment 3/MicroImageNet/validation',
        target_size=input_dim,
        batch_size=batch_val_size,
        class_mode='categorical')


# In[6]:

# create the base pre-trained model
base_model = Xception(weights='imagenet', include_top=False, input_tensor=Input(shape=(256, 256, 3)))


# In[7]:

# add a global spatial average pooling layer
x = base_model.output
x = GlobalAveragePooling2D()(x)
# and a logistic layer -- let's say we have 200 classes
x = Dense(1024, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(200, activation='softmax')(x)

# In[8]:

# this is the model we will train
model = Model(input=base_model.input, output=predictions)

# first: train only the top layers (which were randomly initialized)
# i.e. freeze all convolutional InceptionV3 layers
for layer in base_model.layers:
   layer.trainable = False

model.compile(optimizer='rmsprop', loss='categorical_crossentropy',  metrics=['accuracy'])

model.fit_generator(train_generator,
                    steps_per_epoch=train_size // batch_size,
                    epochs= 2,
                    workers = 4,
                    validation_data=validation_generator,
                    validation_steps=val_size // batch_val_size)

# In[10]:

for layer in model.layers:
    layer.trainable = True



# In[12]:

model.load_weights("weights16_02_0.67.hdf5")


# In[ ]:

sgd = SGD(lr=0.01, momentum=0.7, nesterov=True, decay=5*1e-4)

model.compile(optimizer=sgd, loss='categorical_crossentropy',  metrics=['accuracy'])

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-8)
    

model.fit_generator(train_generator,
                    steps_per_epoch=train_size // batch_size,
                    epochs=epochs,
                    workers = 4,
                    validation_data=validation_generator,
                    validation_steps=val_size // batch_val_size, callbacks=[tb, md, reduce_lr, SGDLearningRateTracker()])


# In[47]:
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
        image *= 1./255

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


classify('/home/balkhamissi/Desktop/Assignment 3/MicroImageNet/new_predictions',
'/home/balkhamissi/Desktop/Assignment 3/MicroImageNet/test_images/',
'/home/balkhamissi/Desktop/Assignment 3/MicroImageNet/wnids.txt',model,256)


print ("Finished!!")