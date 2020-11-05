#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 23 14:05:57 2020

@author: gama
"""
import tensorflow as tf



string= tf.Variable("string Xd",tf.string)


floaing = tf.Variable(3.5645,tf.float64)

rank2_tensor= tf.Variable([["Test","ok","lol"], 
                           ["wtf","hh","ggff"]],tf.string)

#print(tf.rank(rank2_tensor))
#print(rank2_tensor.shape)


tensor1= tf.ones([1,2,3])

#print(tensor1)

tensor2= tf.reshape(tensor1,[2,3,1])

#print(tensor2)



t= tf.zeros([5,5,5,5])
print(t)


t = tf.reshape(t,[125,-1])
print(t)