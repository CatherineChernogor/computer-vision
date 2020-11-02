import matplotlib.pyplot as plt
from skimage.filters import sobel, threshold_isodata
from skimage.morphology import binary_closing, binary_opening
from skimage.measure import label, regionprops
image = plt.imread('lama-on-moon.png')[40:-40, 40:-20]

image = image[:,:, 0]

sobel_mask = sobel(image)
tresh = threshold_isodata(sobel_mask)

sobel_mask[sobel_mask < tresh] =0
sobel_mask[sobel_mask>= tresh] = 1

for i in range(10):
    sobel_mask = binary_closing(sobel_mask)




labeled = label(sobel_mask)

for region in regionprops(labeled):
    if region.area<200 or region.perimeter<1000:
        sobel_mask[region.coords[:, 0], region.coords[:, 1]] = 0
    else: 
        print(region.perimeter)

# plt.hist(tresh.flatten())
# plt.yscale('log')
plt.subplot(121)
plt.imshow(sobel_mask, cmap="gray")
plt.subplot(122)
plt.imshow(image, cmap="gray")
plt.show()