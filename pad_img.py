import os
from glob import glob
import imageio
import numpy as np
import matplotlib.pyplot as plt

# os.chdir('/home/vmlubuntu/Desktop/Just song sequence')
filepath = '/home/vmlubuntu/bird/Birds-no number unpadded' + '/'
savepath= '/home/vmlubuntu/bird/doubled_image/'

files = glob(filepath+'*.png')
# imgs = np.zeros([len(files),256, 1200, 3])
for idx, filename in enumerate(files):
  f = filename.split('/')
  fname = f[-1]
  img = imageio.imread(filename)
  h = 1200 - img.shape[0]
  w = 1200 - img.shape[1]
  copy_number=  int(h/img.shape[0])
  for i in range(copy_number-1):
      if i == 0:
          ims= np.concatenate((img,img),axis=0)
      else:
          ims = np.concatenate((ims, img), axis=0)
  h = 1200 - ims.shape[0]
  hz = np.zeros([h, ims.shape[1], 3])
  lz = np.zeros([1200, w, 3])
  im = np.concatenate((ims, hz), axis=0)
  im = np.concatenate((im, lz), axis=1)
  name = savepath + str(fname)
  imageio.imwrite(name, im.astype(np.uint8))

print('padded images saved to ' + savepath)
