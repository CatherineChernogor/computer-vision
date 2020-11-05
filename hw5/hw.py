# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import morphology
from skimage.measure import label, regionprops
from skimage.filters import try_all_threshold, threshold_triangle


def lakes(image):
    B = ~image
    BB = np.ones((B.shape[0]+2, B.shape[1] + 2))
    BB[1:-1, 1:-1] = B
    return np.max(label(BB)) - 1


def has_vline(image):
    lines = np.sum(image, 0) // image.shape[0]
    # print(lines)
    return 1 in lines


def has_hline(image):
    lines = np.sum(image, 1) // image.shape[1]
    # print(lines)
    return 1 in lines


def has_bay(image):
    b = ~image
    bb = np.zeros((b.shape[0]+1, b.shape[1])).astype('uint8')
    bb[:-1, :] = b
    return lakes(~bb) - 1


def count_bays(image):
    holes = ~image.copy()
    return np.max(label(holes))


def show_symbol(SYM, regions):
    a, b, c = 1, 9, 1
    for region in regions:
        symbol = recognize(region)
        if c == 10:
            c = 1
            a += 1
            plt.show()
        if symbol == SYM:
            case = str(a)+str(b)+str(c)
            plt.subplot(int(case))
            plt.imshow(region.image)
            c += 1
    plt.show()


def recognize(region):
    lc = lakes(region.image)
    if lc == 2:
        if count_bays(region.image) > 4:
            return '8'
        else:
            return 'B'

    if lc == 1:
        # print('0DPA')

        bays = count_bays(region.image)
        if has_vline(region.image):
            if bays > 3:
                return '0'
            else:
                if (region.perimeter**2)/region.area < 58:
                    return 'P'
                else:
                    return 'D'
        else:
            if bays < 5:
                return 'A'
            else:
                return '0'

    if lc == 0:
        # print('XW*/1-')

        bays = count_bays(region.image)

        if has_vline(region.image):

            if np.all(region.image == 1):
                return '-'

            if bays == 5:
                return '*'
            return '1'

        if bays == 2:
            return '/'

        if bays == 5:
            if has_hline(region.image):
                return '*'
            return 'W'

        # left only bays = 4
        if count_bays(region.image[2:-2, 2:-2]) == 5:
            return '*'
        else:
            return 'X'

    return None


image = plt.imread('hw5/symbols.png')

gray = np.sum(image, 2)
gray[gray > 0] = 1

labeled = label(gray)
# print('total', np.max(labeled))

regions = regionprops(labeled)

d = {}
for region in regions:
    symbol = recognize(region)
    if symbol not in d:
        d[symbol] = 1
    else:
        d[symbol] += 1

# for key in d.keys():
#     print(key, '\n')
#     show_symbol(key, regions)

for key in d.keys():
    d[key] = d[key]/np.max(labeled)

print(d)
