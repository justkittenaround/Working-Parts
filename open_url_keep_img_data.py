#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 10 16:56:45 2019

@author: mpcr
"""
import numpy as np
import PIL
from PIL import Image
import requests
from io import BytesIO
from scipy.misc import imread, imresize, bytescale


data0 = np.genfromtxt('csv of urls', delimiter=',', usecols=0, dtype=str)
data0.shape

data1 = np.genfromtxt('csv of urls', delimiter=',', usecols=0, dtype=str)
data1.shape

# make data even number of links (if data1 is larger than data0)
data1 = np.random.choice(data1, data2.shape, replace=False)
data1.shape

imsz = 150 #image size
#create empty sets for the image data to go into
set0 = np.zeros([data0.shape[0], imsz, imsz])
set1 = np.zeros([data1.shape[0], imsz, imsz])

def get_img(datan, setn):
    for idx, url in enumerate (data):
        response = requests.get(url) #open the url
        img = PIL.Image.open(BytesIO(response.content)) #get the image
        img = imresize(img, [imsz, imsz]) #resize the image
        if len(img.shape) == 3: #make images greyscale if they are color
            img = np.mean(img, 2)
        setn[idx, ...] = img #save the images in the sets


get_img(data0, set0)
get_img(data1, set1)




