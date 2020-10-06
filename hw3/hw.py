# -*- coding: utf-8 -*-

import numpy as np
from scipy.ndimage import morphology as mrg

image = np.load('ps.npy.txt')

mask1 = np.array([[1,1,1,1],
                  [1,1,1,1],
                  [0,0,1,1],
                  [0,0,1,1],
                  [1,1,1,1],
                  [1,1,1,1]])

mask2 = np.array([[1,1,1,1],
                  [1,1,1,1],
                  [1,1,0,0],
                  [1,1,0,0],
                  [1,1,1,1],
                  [1,1,1,1]])

mask3 = np.array([[1,1,1,1,1,1],
                  [1,1,1,1,1,1],
                  [1,1,0,0,1,1],
                  [1,1,0,0,1,1]])

mask4 = np.array([[1,1,0,0,1,1],
                  [1,1,0,0,1,1],
                  [1,1,1,1,1,1],
                  [1,1,1,1,1,1]])

mask5 = np.array([[1,1,1,1,1,1],
                  [1,1,1,1,1,1],
                  [1,1,1,1,1,1],
                  [1,1,1,1,1,1]])

masks = np.array([mask1, mask2, mask3, mask4, mask5])

result = []

for i in range(5):
    image_new = mrg.binary_opening(image, masks[i])
    result.append(np.sum(image_new)/np.sum(masks[i]))

symbols = ['ɔ','c','п','ப','࡮']
for i in range(len(result)):
    print(symbols[i], 'symbols -', result[i])
print('all symbols - ',np.sum(result))