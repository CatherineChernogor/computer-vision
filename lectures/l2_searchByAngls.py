import matplotlib.pyplot as plt
import numpy as np


def match(a, masks):
    for mask in masks:
        # if np.all(np.array_equal(a,mask)):
        if np.all(a == mask):
            return True
    return False


def count_objects(image):
    inner, outer, diag = 0, 0, 0
    for y in range(0, image.shape[0]-1):
        for x in range(0, image.shape[1]-1):
            sub = image[y:y+2, x:x+2]
            if match(sub, outer_masks):
                outer += 1
                continue
            if match(sub, inner_masks):
                inner += 1
                continue
            if match(sub, diag_mask):
                diag += 1
    print(f'Innner = {inner}, outer = {outer}')
    return (outer+(diag*2) - inner) / 4


inner_masks = [np.array([[1, 1], [1, 0]]),
               np.array([[1, 1], [0, 1]]),
               np.array([[1, 0], [1, 1]]),
               np.array([[0, 1], [1, 1]])]

outer_masks = [np.array([[0, 0], [0, 1]]),
               np.array([[0, 0], [1, 0]]),
               np.array([[0, 1], [0, 0]]),
               np.array([[1, 0], [0, 0]])]

diag_mask = [np.array([[0, 1], [1, 0]]),
             np.array([[1, 0], [0, 1]])]


if __name__ == "__main__":

    image = np.load('src/cex1npy.txt')

    image = np.sum(image, 2)
    print(count_objects(image))
    plt.imshow(image)

    img = np.load('src/cex2npy.txt')

    for i in range(img.shape[2]):
        print(count_objects(img[:, :, i]))
    plt.imshow(img)
