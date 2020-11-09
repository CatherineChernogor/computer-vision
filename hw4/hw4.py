import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import morphology
from skimage.measure import label, regionprops
from skimage.filters import threshold_triangle


def togray(image):
    return (0.2989*image[:, :, 0]+0.587*image[:, :, 1]+0.114 * image[:, :, 2]).astype('uint8')


def binarisation(image, limit_min, limit_max):
    B = image.copy()
    B[B <= limit_min] = 0
    B[B >= limit_max] = 0
    B[B > 0] = 1
    return B


def circularity(region, label=1):
    return (region.perimeter**2)/region.area


def get_pencils(filename):
    image = plt.imread(filename)
    gray = togray(image)

    thresh = threshold_triangle(gray)
    binary = binarisation(gray, 0, thresh)
    binary = morphology.binary_erosion(binary, iterations=10)
    binary = morphology.binary_dilation(binary, iterations=10)
    labeled = label(binary)

    areas = []
    for region in regionprops(labeled):
        areas.append(region.area)

    for region in regionprops(labeled):
        if region.area < np.mean(areas):
            labeled[labeled == region.label] = 0
        bbox = region.bbox
        if bbox[0] == 0 or bbox[1] == 0:
            labeled[labeled == region.label] = 0

    labeled[labeled > 0] = 1
    labeled = label(labeled)

    i, count = 1, 0
    for region in regionprops(labeled):
        isCirc = circularity(region, i)
        if isCirc > 100 and region.area < 450000 and region.area > 300000:
            count += 1
        i += 1
    return count


total = 0
for i in range(1, 13):
    total += get_pencils('hw4/img ('+str(i)+').jpg')
    
print('total:', total)
