#!/usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import face

def translate(image, vector):
    tlanslated = np.zeros_like(image)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            
            ni = i - vector[0]
            nj = j - vector[1]
            
            if ni < 0 or nj < 0:
                continue
                
            if ni >= image.shape[0] or nj >= image.shape[1]:
                continue
                
            tlanslated[ni, nj] = image[i, j]
    return tlanslated


if __name__ == "__main__":
    raccon = face(True)
    translated = translate(raccon, (-250, 450))

    plt.figure()
    plt.subplot(131)
    plt.imshow(raccon, cmap='magma')
    plt.subplot(132)
    plt.imshow(translated, cmap='magma')


