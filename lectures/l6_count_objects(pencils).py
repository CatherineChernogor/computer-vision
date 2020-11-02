# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import morphology
from skimage.measure import label, regionprops
from skimage.filters import try_all_threshold, threshold_triangle


def togray(image):
    return (0.2989*image[:, :, 0]+0.587*image[:, :, 1]+0.114 * image[:, :, 2]).astype('uint8')


def binarisation(image, limit_min, limit_max):
    B = image.copy()
    B[B <= limit_min] = 0
    B[B >= limit_max] = 0
    B[B > 0] = 1
    return B


image = plt.imread('img (8).jpg')
plt.imshow(image)
plt.show()

gray = togray(image)
plt.imshow(gray, cmap="gray")
plt.show()

thresh = threshold_triangle(gray)
binary = binarisation(gray, 0, thresh)
# binary = morphology.binary_erosion(binary, iterations=10)
binary = morphology.binary_dilation(binary, iterations=5)

# plt.imshow(binary)

labeled = label(binary)
# plt.imshow(labeled)
print(np.max(labeled))

areas = []
for region in regionprops(labeled):
    areas.append(region.area)

print(np.mean(areas))

for region in regionprops(labeled):
    if region.area < np.mean(areas):
        labeled[labeled == region.label] = 0
    bbox = region.bbox
    if bbox[0] == 0 or bbox[1] == 0:
        labeled[labeled == region.label] = 0

labeled[labeled > 0] = 1
labeled = label(labeled)

plt.imshow(labeled)
plt.show()

print(np.max(labeled))


def hist(image):
    H = np.zeros(256)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            val = image[i, j]
            H[val] += 1
    return H


H = hist(gray)
plt.plot(H)

tresh = threshold_triangle(gray)
print(tresh)

try_all_threshold(gray, figsize=(12, 10), verbose=False)
