# Задание: Определить смещение

import numpy as np
import os
import matplotlib.pyplot as plt


def read_img(filename):
    temp = []
    lines = open(filename).readlines()
    for line in lines[2:]:
        temp.append(line.split())

    return np.array(temp, dtype='int32')

img1 = read_img('img1.txt')
img2 = read_img('img2.txt')

def find_first_point(img):
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            if img[y, x] == 1:
                return y, x

y1, x1 = find_first_point(img1)
y2, x2 = find_first_point(img2)

print(f'сдвиг по оси у = {y2-y1} и по оси х = {x2-x1}')