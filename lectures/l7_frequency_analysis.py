# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import morphology
from skimage.measure import label, regionprops
from skimage.filters import try_all_threshold, threshold_triangle

image = plt.imread('lectures/src/alphabet.png')
# plt.figure(figsize=(12, 12))
# plt.imshow(image)
# plt.show()

gray = np.sum(image, 2)
# print(np.min(gray),np.max(gray))

gray[gray > 0] = 1


labeled = label(gray)

# plt.figure(figsize=(25, 25))
# plt.imshow(labeled)
# plt.show()
print(np.max(labeled))

regions = regionprops(labeled)

plt.imshow(regions[121].image)


def lakes(image):
    B = ~image
    BB = np.ones((B.shape[0]+2, B.shape[1] + 2))
    BB[1:-1, 1:-1] = B
    return np.max(label(BB)) - 1


def has_vline(image):
    lines = np.sum(image, 0) // image.shape[0]
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
            plt.subplot(str(a)+str(b)+str(c))
            plt.imshow(region.image)
            c += 1
    plt.show()


def recognize(region):
    lc = lakes(region.image)
    if lc == 2:
        if has_vline(region.image):
            return "B"
        return "8"
    if lc == 1:

        if has_bay(region.image) > 0:
            return "A"
        return "0"
    if lc == 0:
        if has_vline(region.image):
            if np.all(region.image == 1):
                return '-'
            return '1'

        bays = count_bays(region.image)
        if bays == 2:
            return '/'

        if bays > 3:
            # print("W or X or *")
            circ = region.perimeter**2 / region.area
            if circ > 70:
                return '*'
            if bays == 5:
                return 'W'
            if bays == 4:
                return 'X'

    return None


d = {}
for region in regions:
    symbol = recognize(region)
    if symbol not in d:
        d[symbol] = 1
    else:
        d[symbol] += 1
print(d)

show_symbol('8', regions)