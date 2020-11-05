#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ref:
    https://colab.research.google.com/drive/1ZZXnCjFEOkp_KdNcNabd14yok0BAIuwS#forceEdit=true&sandboxMode=true
"""


# CNN Tuto
import tensorflow as tf
from tensorflow import keras
#from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import numpy as np 

#%%BLOCK 1: DATA LOAD
print("Loading data")
#  LOAD AND SPLIT DATASET
(train_images, train_labels), (test_images, test_labels) = keras.datasets.cifar10.load_data()

# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']
#%% VIEW DATA

# Let's look at a one image
IMG_INDEX = 77 # change this to look at other images

plt.imshow(train_images[IMG_INDEX] ,cmap=plt.cm.binary)
plt.xlabel(class_names[train_labels[IMG_INDEX][0]])
plt.show()

#%% MODEL
print("Adding layers to the model...")
model = keras.Sequential()
model.add(keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(keras.layers.MaxPooling2D((2, 2)))
model.add(keras.layers.Conv2D(64, (3, 3), activation='relu'))
model.add(keras.layers.MaxPooling2D((2, 2)))
model.add(keras.layers.Conv2D(64, (3, 3), activation='relu'))
"""
Layer 1

The input shape of our data will be 32, 32, 3 and we will process 32 filters of size 3x3 over our input data. W
e will also apply the activation function relu to the output of each convolution operation.

Layer 2

This layer will perform the max pooling operation using 2x2 samples and a stride of 2.

Other Layers

The next set of layers do very similar things but take as input the feature map from the previous layer. 
They also increase the frequency of filters from 32 to 64. We can do this as our data shrinks 
in spacial dimensions as it passed through the layers, meaning we can afford (computationally) 
to add more depth.

"""

#%% Model summary
model.summary()

#%%ADD DENSE LAYERS (classifier)
    
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(64, activation='relu'))
model.add(keras.layers.Dense(10))

#%%new mode summary
model.summary()

#%% TRAININg
print("Compiling...")
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

print("Starting training...")
history = model.fit(train_images, train_labels, epochs=10, 
                    validation_data=(test_images, test_labels))

#%% TEST SET
test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
print(test_acc)


#%% PREDICTIONS

#class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
#               'dog', 'frog', 'horse', 'ship', 'truck']


#CHOOSE INDEX TO TEST
print("ENTER A NUMBER INt:")
index=int(input())
predictions = model.predict(test_images) # ARRAY OF PREDS FOR TEST IMAGE

imageTest= test_images[index]
labelTest= test_labels[index] #True label

print("The prediction is:",np.argmax(predictions[index]))
print("The true label is:",labelTest)
plt.figure()
plt.imshow(test_images[index])
plt.colorbar()
plt.show()
plt.title("True label: "+str(labelTest)+"  VS  the prediction: "+str(np.argmax(predictions[index])))






#%% SAVE model

model.save('myModel_Cifar10CNN.h5')












