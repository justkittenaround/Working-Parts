import os
from glob import glob
import imageio
import numpy as np
import matplotlib.pyplot as plt

img_path = '/home/vmlubuntu/bird/doubled_image'
save_path = '/home/vmlubuntu/bird/stitched_double_image/'


#sorting_ *this is done.. official sorted list = master______________________________________________________________________
os.chdir(img_path)
names = sorted(glob('*.png'))
names.sort()
num_names = len(names)
bin = []
number = []
folders = []
for file in names:
  split_name = file.split('-')
  folder = split_name[0]
  folders.append(folder)
  filename = split_name[1]
  bin.append(filename)
  filename = filename.split('.')
  filename = filename[0]
  number.append(filename)

number.sort(key=float)
songs = []
for song in number:
    file = str(song) + '.png'
    songs.append(file)

master = []
for song in songs:
    idx = (bin.index(song))
    name = str(folders[idx]) + '-' + song
    master.append(name)


#stacking___(for pix2pix)___________________________________________________________________
import cv2
images= []
for file in master:
    a= cv2.imread(file)
    images.append(a)

for i,image in enumerate(images):
    a= images[i]
    try:
        b= images[i+1]
    except:
        print('Done!')
        break
    c= np.concatenate((a,b),axis=1)
    name = save_path+str(i)+'-'+str(i+1)+'.png'
    cv2.imwrite(name, c)




#seperate into train and validate-----------------------------------------------------------
Two_image='/home/vmlubuntu/bird/stitched_double_image/train' + '/'
val_image='/home/vmlubuntu/bird/stitched_double_image/val' + '/'
All_image=os.listdir(Two_image)
number=len(All_image)
val_number=round(number*.2)
song_choice=np.random.choice(All_image,size=val_number,replace=False)
import shutil
for song in song_choice:
    src= Two_image+song
    dst=val_image + song
    shutil.move(src,dst)
