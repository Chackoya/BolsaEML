#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 24 20:53:06 2020

@author: gama
"""



# MNIST FASHION TESTING TUTORIAL 
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
import math

#%%Block1: DATA SETS

fashion_mnist = keras.datasets.fashion_mnist  # load dataset
 # split into tetsing and training
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data() 



#%%BLOCK 2: PREVIEW

print(train_images.shape) # (60000, 28, 28)

print(train_images[0,23,23]) #VIEW ONE PIXEL. value between 0 and 255.

print(train_labels[:10]) # let's have a look at the first 10 training labels


#%%BLOCK 3: VIEW DATA
def affiche_imagesMNIST(data,labels,nb_img):
    sqr= math.sqrt(nb_img)
    plt.figure(figsize=(15,15))
    for i in range(nb_img):#affiche 25 images  et leurs labels
        plt.subplot(int(sqr)+1,int(sqr)+1,i+1) # 5x5 l'affichage
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        #plt.colorbar()
        plt.imshow(data[i].reshape(28,28),cmap=plt.cm.binary) #remove cmap-> no color
        plt.xlabel(labels[i])
    plt.show()


affiche_imagesMNIST(train_images,train_labels,10)


#%% DATA PREPROCESSING: scale all our greyscale pixel values (0-255) to be between 0 and 1
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
train_images = train_images / 255.0

test_images = test_images / 255.0

#%% BUILDING THE MODEL -architecture of the neural network.

"""
using Keras sequential model with three different layers. 
This model represents a feed-forward neural network (one that passes values from left to right)
"""
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),  # input layer (1)
    keras.layers.Dense(128, activation='relu'),  # hidden layer (2)
    keras.layers.Dense(10, activation='softmax') # output layer (3)
])

"""
Layer 1: This is our input layer and it will conist of 784 neurons.
 We use the flatten layer with an input shape of (28,28) to denote that our input should come in in that shape. 
 The flatten means that our layer will reshape the shape (28,28) 
 array into a vector of 784 neurons so that each pixel will be associated with one neuron.


Layer 2:This is our first and only hidden layer.
 The dense denotes that this layer will be fully connected and each neuron 
 from the previous layer connects to each neuron of this layer. 
 It has 128 neurons and uses the rectify linear unit activation function.


"""
#%%COMPILE THE MODEL

#hyper parameters can be changed.
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


#%%TRAINING
model.fit(train_images, train_labels, epochs=15)  #Accuracy training: 91% with 10 epochs ; 94% with 15epochs. mby overfit

#%%TESTING DATA
test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=1) 

print('Test accuracy:', test_acc)


#RESULT TEST: 88%



#%% MAKE PREDICTIONS
index=5
predictions = model.predict(test_images)
print(class_names[np.argmax(predictions[index])])
plt.figure()
plt.imshow(test_images[index])
plt.colorbar()
plt.show()
plt.title(class_names[test_labels[index]])


#%%VERIFY PREDS
COLOR = 'white'
plt.rcParams['text.color'] = COLOR
plt.rcParams['axes.labelcolor'] = COLOR

def predict(model, image, correct_label):
  class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
  prediction = model.predict(np.array([image]))
  predicted_class = class_names[np.argmax(prediction)]

  show_image(image, class_names[correct_label], predicted_class)


def show_image(img, label, guess):
  plt.figure()
  plt.imshow(img, cmap=plt.cm.binary)
  #plt.title("Excpected: " + label)
  #plt.xlabel("Guess: " + guess)
  print("Expected label:",label)
  print("Expected guess:",guess)
  plt.colorbar()
  plt.grid(False)
  plt.show()


def get_number():
  while True:
    num = input("Pick a number: ")
    if num.isdigit():
      num = int(num)
      if 0 <= num <= 1000:
        return int(num)
    else:
      print("Try again...")

num = get_number()
image = test_images[num]
label = test_labels[num]
predict(model, image, label)





#%% Save model

model.save('myModelFASHION_MNIST.h5')

















