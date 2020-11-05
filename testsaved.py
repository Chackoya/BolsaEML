#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 25 00:44:55 2020

@author: gama
"""
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
fashion_mnist = keras.datasets.fashion_mnist  # load dataset
 # split into tetsing and training
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data() 

train_images = train_images / 255.0

test_images = test_images / 255.0

newmodel= keras.models.load_model('myModelFASHION_MNIST.h5')

newmodel.summary()

#%%
loss, acc= newmodel.evaluate(test_images,test_labels,verbose=1)


preds= newmodel.predict(test_images)


class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
#print(np.argmax(preds[1]))

#%% preds
print("ENTER A NUMBER INT BETWEEN 0 AND 10K:")
index=int(input())
predictions = newmodel.predict(test_images) # ARRAY OF PREDS FOR TEST IMAGE

imageTest= test_images[index]
labelTest= test_labels[index] #True label

print("The prediction is:",np.argmax(predictions[index]))
print("The true label is:",labelTest)
plt.figure()
plt.imshow(test_images[index])
plt.colorbar()
plt.show()
plt.title("True label: "+str(labelTest)+"  VS  the prediction: "+str(np.argmax(predictions[index])))
