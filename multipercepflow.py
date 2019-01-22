#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 13 19:02:04 2018

@author: mpcr
"""
#perceptron#
 #############################
    
#import utilities
import matplotlib.pyplot as plt
import os, sys
import numpy as np
from scipy.misc import bytescale
import progressbar

#get data
data = np.genfromtxt('linear_data.csv', delimiter=',') #1k rows; 3 columns: 1:x 2:y 3:0 or 1
print(data.shape)

#assign variables x and y to data
labels = data[:, -1]
data = data[;, ;-1]
print(data.shape)
print(labels.shape)

#visualize data
plt.scatter(data[labels == 1, 0], data[labels == 1, 1], c='r')
plt.scatter(data[labels == 0, 0], data[labels == 0, 1], c='b')
plt.show()

#assign arbitrary 1's for y-intercept bias to change as weights change
biases = np.ones([data.shape[0], 1])
data = np.concatenate((data, biases), 1)
print(data.shape)

#define weights for x, y, and y-intercept
weights = np.random.randn(3, )
print(weights)

#set hyperparameter: learning rate
lr = 0.05

#create visualization tool to see each iterations error
se = [] #empy list

#initialize progressbar
bar = progressbar.ProgressBar()
 
#build network--> nonlinearly activating nodes fully connected in each layer   
    #define input layer  
    #define hidden layer (one or more)
    #define output layer
        #activation function
for i in bar(range(1000)):
        x = data[i, :]
        y = labels[i]
        out = np.dot(x, weights)
        se.append((y - out)**2)
        error = y - np.round(out)
        x = error * x
        weights += lr * x
        
#train
    #stochasitc gradient descent
        #backpropagation to convergence
all_out = np.round(np.matmul(data, weights))
acc = np.mean(all_out == labels)
print(acc)
print(weights)
   
#visualize training data
pred_neg = data[all_out == 0]
pred_pos = data[all_out == 1]     
diff = labels - all_out
wrong_pred = data[diff != 0, :]
fig = plt.figure(figsize=(12, 6))
subplot1 = fig.add_subplot(131)
subplot2 = fig.add_subplot(132)
subplot3 = fig.add_subplot(133)
subplot1.set_xlabel('training iteration')
subplot1.set_ylabel('Error Squared')
subplot1.plot(np.absolute(se))
subplot2.set_title('Classification based on learned weights')
subplot2.scatter(pred_pos[:, 0], pred_pos[:, 1], c='r')
subplot2.scatter(pred_neg[:, 0], pred_neg[:, 1], c='b')
subplot3.set_title('Incorrectly Classified Points')
subplot3.scatter(wrong_pred[:, 0], wrong_pred[:, 1], c='g')
plt.tight_layout()

#validate trained model with new data
test_data = np.random.rand(1000, 2)
test_data = np.concatenate((test_data, biases), 1)
test_out = np.round(np.matmul(test_data, weights))
test_pos = test_data[test_out == 1, :]
test_neg = test_data[test_out == 0, :]
plt.scatter(test_pos[:, 0], test_pos[:, 1], c='r')
plt.scatter(test_neg[:, 0], test_neg[:, 1], c='b')
plt.show()

###############################
#multilayer perceptron#
    #supervised learning
    #binary classifyer
 #############################