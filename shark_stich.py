import os
import numpy as np
from PIL import Image



og_path = 'Desktop/unlabeled original pair/'
labled_path = 'Desktop/Labeled shark image/'
save_path = 'pytorch-CycleGAN-and-pix2pix/datasets/sharks/train/'

og = os.listdir(og_path)
labled = os.listdir(labled_path)

pad = np.zeros((416,416,1))

for idx, f in enumerate(og):
  im = Image.open(og_path+f)
  im = np.asarray(im)
  im = np.append(im, pad, axis=2)
  im2 = Image.open(labled_path + labled[idx])
  im2 = np.asarray(im2)
  c = np.append(im, im2, axis=1)
  cv2.imwrite(save_path + f, c)
