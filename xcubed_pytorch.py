from __future__ import division, print_function, absolute_import
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as f
from torch.autograd import Variable
import h5py
import glob
import sys, os
from torchvision.utils import make_grid
from torchvision.transforms import Pad
from sklearn.preprocessing import scale
from scipy.misc import imresize, bytescale
from skimage.util import view_as_windows as vaw
import time
import cv2
from progressbar import ProgressBar

# define useful variables
iters = 350 # number of training iterations
batch_sz = 100  # training batch size
k = 500
patch_sz = 15

fnames = glob.glob('*.h5')


def mat2ten(X):
    zs=[X.shape[1], int(np.sqrt(X.shape[0])), int(np.sqrt(X.shape[0]))]
    Z=np.zeros(zs)

    for i in range(X.shape[1]):
        Z[i, ...] = X[:,i].reshape([zs[1],zs[2]])

    return Z


def montage(X):
    count, m, n = np.shape(X)
    mm = int(np.ceil(np.sqrt(count)))
    nn = mm
    M = np.zeros((mm * m, nn * n))

    image_id = 0
    for j in range(mm):
        for k in range(nn):
            if image_id >= count:
                break
            sliceM, sliceN = j * m, k * n
            M[sliceM:sliceM + m, sliceN:sliceN + n] = bytescale(X[image_id, ...])
            image_id += 1

    return np.uint8(M)


def plot(x):
    fig, ax = plt.subplots()
    im = ax.imshow(x, cmap='gray')
    ax.axis('off')
    fig.set_size_inches(18, 18)
    plt.show()



def file_get(filename):
    f = h5py.File(filename, 'r')
    X = np.asarray(f['X'])
    f.flush()
    f.close()
    return X



def whiten(X):
    '''Function to ZCA whiten image matrix.'''
    U,S,V = torch.svd(torch.mm(X, torch.t(X)))
    epsilon = 1e-5
    ZCAMatrix = torch.mm(U, torch.mm(torch.diag(1.0/torch.sqrt(S + epsilon)),
                                                torch.t(U)))
    return torch.mm(ZCAMatrix, X)



def X3(y, iters, batch_sz, num_dict_features=None, D=None):
    ''' Dynamical systems neural network used for sparse approximation of an
        input vector.

        Args:
            y: input signal or vector, or multiple column vectors.
            num_dict_features: number of dictionary patches to learn.
            iters: number of LCA iterations.
            batch_sz: number of samples to send to the network at each iteration.
            D: The dictionary to be used in the network.'''

    D = torch.randn(patch_sz**2, k).float().cuda(0)
    D = Variable(D, requires_grad=False, volatile=True)

    for i in range(iters):
        # choose random examples this iteration
        x = y[np.random.randint(0, y.shape[0], batch_sz), ...]
        x = torch.from_numpy(x).float().cuda(0)
        x = Variable(x, requires_grad=False, volatile=True)

        x = x.unfold(2, patch_sz, 5).unfold(3, patch_sz, 8).contiguous()
        x = x.view(x.size(0)*x.size(1)*x.size(2)*x.size(3),
                   x.size(-2)*x.size(-1))
        x = whiten(torch.t(x - torch.mean(x, 0)))
        x2 = x[:, x.size(1)//2:].cuda(1)
        x = x[:, :x.size(1)//2]

        D = torch.mm(D, torch.diag(1./(torch.sqrt(torch.sum(D**2, 0))+1e-6)))
        D2 = D.cuda(1)

        a = torch.mm(torch.t(D), x).cuda(0)
        a2 = torch.mm(torch.t(D2), x2).cuda(1)

        a = torch.mm(a, torch.diag(1./(torch.sqrt(torch.sum(a**2, 0))+1e-6)))
        a2 = torch.mm(a2, torch.diag(1./(torch.sqrt(torch.sum(a2**2, 0))+1e-6)))

        a = .8 * a ** 3
        a2 = .8 * a2 ** 3

        x = x - torch.mm(D, a)
        x2 = x2 - torch.mm(D2, a2)

        x = torch.cat((x, x2), 1).cuda(0)
        a = torch.cat((a, a2), 1).cuda(0)

        D = D + torch.mm(x, torch.t(a))

        cv2.namedWindow('dictionary', cv2.WINDOW_NORMAL)
        cv2.imshow('dictionary', montage(mat2ten(D.data.cpu().numpy())))
        cv2.waitKey(1)

    return D.data.cpu().numpy(), a.data.cpu().numpy()



X = file_get(fnames[0])

Bar = ProgressBar()
for i in Bar(range(X.shape[0])):
    X[i, :240//3, :320//3, :] = imresize(X[i, ...], [240//3, 320//3])
X = X[:, :240//3, :320//3, :]

X = np.transpose(np.mean(X, 3, keepdims=True), (0, 3, 1, 2))

d, a = X3(X, iters, batch_sz, k)


#x = torch.ones(4, 3, 6, 7).cuda()
#d = torch.zeros(10, 3, 2, 7).cuda()
#a = f.conv2d(Variable(x), Variable(d)).cuda()
