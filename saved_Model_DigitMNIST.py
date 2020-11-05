#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
'myModel_DigitMNIST.h5' contains the pretrained model

Experiment here


"""
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import math
#%% CELL BLOCK 1: LOAD DATA & PREPROCESS
mnist = tf.keras.datasets.mnist

#Split data into training sets & test sets
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()


train_images = train_images / 255.0

test_images = test_images / 255.0
#%% CELL BLOCK 2: LOAD PRETRAINED MODEL

newmodel= keras.models.load_model('myModel_DigitMNIST.h5')

newmodel.summary()

loss, acc= newmodel.evaluate(test_images,test_labels,verbose=1)


#%% EXPERIMENT  PREDICTIONS

#CHOOSE INDEX TO TEST
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




#%% SHOW MULTIPLE PREDS

#this will plot the images from index 0 to index nb_img (not too big because it will mess up the plot, like 50 max).

def show_imagesMNIST(data,labels,nb_img,pred):
    
    sqr= math.sqrt(nb_img)
    plt.figure(figsize=(15,15))
    for i in range(nb_img):#affiche 25 images  et leurs labels
        plt.subplot(int(sqr)+1,int(sqr)+1,i+1) # 5x5 l'affichage
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(data[i].reshape(28,28))#cmap=plt.cm.binary) #remove cmap-> no color
        #plt.colorbar()
        plt.xlabel("Label:"+str(labels[i])+" Pred:"+str(np.argmax(pred[i])))
    plt.show()

#def show_imagesRange(data,labels, a, b,pred)


show_imagesMNIST(test_images,test_labels,50, predictions)

