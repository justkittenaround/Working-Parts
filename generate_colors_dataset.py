#make colored images

import numpy as np



num_samps = 300
h =
w =
chans = 3


c = np.ones([num_samps, chans, h, w])

cdark = np.zeros([num_samps, chans, h, w])
for idx in range(num_samps):
    cdark[idx, ...] = 0.3

clight = np.zeros([num_samps, chans, h, w])
for idx in range(num_samps):
    clight[idx, ...] = 0.7
