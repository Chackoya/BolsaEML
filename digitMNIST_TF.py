#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DIGIT CLASSIFICATION with Tensorflow

MNIST DATASET - 60k training images and 10k test images 

Reference:
    https://towardsdatascience.com/solve-the-mnist-image-classification-problem-9a2865bcf52a
    tensorflow tutorials

"""
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import math

#%% CELL BLOCK 1: LOAD DATA

mnist = tf.keras.datasets.mnist

#Split data into training sets & test sets
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

#%% CELL BLOCK 2: VISUALIZE DATA 
#print(train_images.shape) # => (60000, 28, 28)


def show_imagesMNIST(data,labels,nb_img):
    sqr= math.sqrt(nb_img)
    plt.figure(figsize=(15,15))
    for i in range(nb_img):#affiche 25 images  et leurs labels
        plt.subplot(int(sqr)+1,int(sqr)+1,i+1) # 5x5 l'affichage
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(data[i].reshape(28,28))#cmap=plt.cm.binary) #remove cmap-> no color
        plt.colorbar()
        plt.xlabel(labels[i])
    plt.show()

#show_imagesMNIST(train_images , train_labels, 10)



#%% CELL BLOCK 3: PREPROCESS DATA
#Scale all values from [0;255] to [0;1] 


train_images = train_images / 255.0
test_images = test_images / 255.0


#%% CELL BLOCK 3: NETWORK MODEL ARCHITECTURE -> FEEDFORWARD NEURAL NETWORK

"""
Layer 1: input layer that consists on 784 neurons;
    - Usage of flatten layer with input shape of (28,28) to denote that our input should come in that shape.
    - " The flatten means that our layer will reshape the shape (28,28) 
 array into a vector of 784 neurons so that each pixel will be associated with one neuron."

Layer 2: the only Hidden layer 
    - Dense means that it's fully connected
    - 128 neurons and usage of rectify linear unit activation fct;
    
Layer 3: Output layer (10 neurons / 10 different classes,digits)
    - usage of softmax (array size 10 with probabilities for each class)
"""

# "Sequential layers dynamically adjust the shape of input to a layer based the out of the layer before it"
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),  # input layer (1)
    tf.keras.layers.Dense(128, activation='relu'),  # hidden layer (2)
    tf.keras.layers.Dense(10, activation='softmax') # output layer (3)
])


#%% CELL BLOCK 4: COMPILATION OF THE MODEL

#Define the loss function, optimizer and metrics we want to see(accuracy)
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


#%% CELL BLOCK 5: TRAINING THE MODEL 

#fit the model to the training data
model.fit(train_images, train_labels, epochs=10)
# => Result for accuracy on training set: 0.9950



#%% CELL BLOCK 6: TESTING THE MODEL

test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=1) 
print('Test accuracy:', test_acc)

# => Result 0.9775 accuracy For test set

#%% CELL BLOCK 7: MAKE PREDICTIONS

#CHOOSE INDEX TO TEST
print("ENTER A NUMBER INT BETWEEN 0 AND 10K:")
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


#%% CELL BLOCK 8: SAVE MODEL FOR LATER USAGE

model.save('myModel_DigitMNIST.h5')









