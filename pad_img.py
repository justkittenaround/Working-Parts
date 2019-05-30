import os
from glob import glob
import imageio
import numpy as np
import matplotlib.pyplot as plt

os.chdir('/home/vmlubuntu/bird/birdsong')
filepath = '/home/vmlubuntu/bird/paddedimgs/'


files = glob('**/*.png', recursive=True)
# imgs = np.zeros([len(files),256, 1200, 3])
for idx, filename in enumerate(files):
  img = imageio.imread(filename)
  h = 256 - img.shape[0]
  w = 1200 - img.shape[1]
  hz = np.zeros([h, img.shape[1], 3])
  lz = np.zeros([256, w, 3])
  im = np.concatenate((img, hz), axis=0)
  im = np.concatenate((im, lz), axis=1)
  # imgs[idx, ...] = im.astype(np.uint8)
  split_name = filename.split('/')
  filename = split_name[1]
  name = filepath + str(filename)
  imageio.imwrite(name, im.astype(np.uint8))

print('padded images saved to ' + filepath)
