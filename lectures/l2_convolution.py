from scipy.misc import face
import numpy as np
import matplotlib.pyplot as plt


def convolve(image, mask):
    image_bounded = np.zeros((image.shape[0]+2, image.shape[1]+2))
    image_bounded[1:-1, 1:-1] = image
    result = np.zeros_like(image)
    for y in range(1, image_bounded.shape[0] - 1):
        for x in range(1, image_bounded.shape[1] - 1):
            sub = image_bounded[y-1:y+2, x-1: x+2]
            new_value = np.sum(sub * mask)  # /np.sum(mask)
            result[y-1, x-1] = new_value
    return result


if __name__ == "__main__":

    image = face(True).astype('f4')
    plt.imshow(image, cmap="gray")

    mask = np.array([[-1, 2, -1], [2, 2, 2], [-1, 2, -1]])
    plt.imshow(convolve(image, mask), cmap='gray')
