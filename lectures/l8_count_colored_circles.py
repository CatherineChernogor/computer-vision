import matplotlib.pyplot as plt
from skimage.filters import sobel, threshold_isodata
from skimage.morphology import binary_closing, binary_opening
from skimage.measure import label, regionprops
from skimage import color
import numpy as np


image = plt.imread('lectures/src/ballss.png')
# image = image[:, :, :-1]
binary = image.copy()[:, :, 0]
binary[binary > 0] = 1

image = color.rgb2hsv(image)[:, :, 0]

labeled = label(binary)
print('total=', np.max(labeled))

colors = []

for region in regionprops(labeled):
    bb = region.bbox
    val = np.max(image[bb[0]:bb[2], bb[1]:bb[3]])
    colors.append(val)

colors.sort()


plt.figure()
plt.plot(np.diff(colors), 'o')

plt.figure()
plt.imshow(image)
plt.show()
