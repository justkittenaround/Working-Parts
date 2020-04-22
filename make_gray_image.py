


folder = 'Documents/bird/Birds-no number unpadded' + '/'
save_folder = 'Documents/bird/grey_birds_no_scale_unstiched' + '/'

import os
import numpy as np
from PIL import Image

# sub_folders = os.listdir(folder)

# for fol in sub_folders:

for file in os.listdir(folder):
        if '.png' in file:
            im = Image.open(folder  + file)
            image = im.convert(mode='L')
            name = file.split('.')
            name = name[0]
            name = name + '.jpg'
            image.save(save_folder+name, 'JPEG')
