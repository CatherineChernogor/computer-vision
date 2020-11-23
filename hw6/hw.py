import matplotlib.pyplot as plt
from skimage.filters import sobel, threshold_isodata
from skimage.morphology import binary_closing, binary_opening
from skimage.measure import label, regionprops
from skimage import color
import numpy as np


def circularity(region):
    return (region.perimeter**2)/region.area

def sides(bbox):
    return bb[2] - bb[0], bb[3] - bb[1]


def count_figs(colors, name):
    colors.sort()
    # colors = [round(circ, 2) for circ in colors]
    # plt.figure()
    # plt.plot(np.diff(colors), 'o')
    # plt.show()
    color_set = {}
    for color in colors:
        if len(color_set) > 0:
            last = list(color_set.keys())[-1]

            if color != last:
                if color > last + 0.02:
                    cur_key = color
                    color_set[cur_key] = 1
                else:
                    color_set[cur_key] += 1
            else:
                color_set[cur_key] += 1

        else:
            cur_key = color
            color_set[cur_key] = 1

    print(f'{name}: {color_set}')


image = plt.imread('hw6/balls_and_rects.png')

binary = image.copy()[:, :, 0]
binary[binary > 0] = 1

image = color.rgb2hsv(image)[:, :, 0]

labeled = label(binary)
print('total:', np.max(labeled))



colors_rect, colors_circ = [], []
for region in regionprops(labeled):
    bb = region.bbox
    val = np.max(image[bb[0]:bb[2], bb[1]:bb[3]])

    a, b = sides(bb)
    # print(val)
    if a * b == region.area:
        colors_rect.append(val)
    else:
        colors_circ.append(val)


count_figs(colors_circ, 'circle')
count_figs(colors_rect, 'rectangle')
