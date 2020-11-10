import matplotlib.pyplot as plt
from skimage.filters import sobel, threshold_isodata
from skimage.morphology import binary_closing, binary_opening
from skimage.measure import label, regionprops
from skimage import color
import numpy as np
import math


def circularity(region):
    return (region.perimeter**2)/region.area

def sides(bbox):
    return bb[2] - bb[0], bb[3] - bb[1]


def count_figs(colors, name):
    colors.sort()
    colors = [round(circ, 2) for circ in colors]

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

    print(name, color_set)


image = plt.imread('hw6/balls_and_rects.png')
binary = image.copy()[:, :, 0]
binary[binary > 0] = 1

image = color.rgb2hsv(image)#[:, :, 0]

labeled = label(binary)
print('total:', np.max(labeled))
plt.figure()
plt.imshow(image[:, :, 0])
plt.figure()
plt.imshow(image[:, :, 1])
plt.figure()
plt.imshow(image[:, :, 2])
plt.show()

# colors_rect, colors_circ = [], []
# for region in regionprops(labeled):
#     bb = region.bbox
#     val = np.max(image[bb[0]:bb[2], bb[1]:bb[3]])

#     a, b = sides(bb)
#     if a * b == region.area:
#         colors_rect.append(val)
#     else:
#         colors_circ.append(val)


# count_figs(colors_circ, 'circle')
# count_figs(colors_rect, 'rectangle')

# colors_style = {
#     0.05: "red", 0.19: "yellow", 0.3: 'cyan',
#     0.42: 'magenta', 0.61: 'brown', 0.83: 'orange'
# }
