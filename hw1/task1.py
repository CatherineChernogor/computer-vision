# Задание: Номинальное разрешение

import numpy as np
import os
import matplotlib.pyplot as plt

filename = 'figure'
ending = '.txt'

for i in range(1,7):

    file = open(filename+str(i)+ending)
    width_mm = float(file.readline())
    lines = file.readlines()[1:]
    file.close()

    len_img = len(lines[0].split())

    resolution = width_mm/len_img
    print(f'Resolution for figure{i} = {resolution}, \n\tmax width in mm = {width_mm}, \t pixels width = {len_img}\n\n')
