from skimage.measure import label
import numpy as np
import matplotlib.pyplot as plt


def area(img, label=1):
    area = np.sum(img[img == label])
    return area/label


def centroid(img, label=1):
    pos = np.where(img == label)
    cy = np.mean(pos[0])
    cx = np.mean(pos[1])
    return cy, cx


def neighbors(y, x):
    return ((y, x+1), (y+1, x), (y, x-1), (y-1, x))


def get_boundaries(img, label=1):
    pos = np.where(img == label)
    boundaries = []
    for y, x in zip(*pos):
        for yn, xn in neighbors(y, x):
            if yn < 0 or yn > img.shape[0] - 1:
                boundaries.append((y, x))
                break

            if xn < 0 or xn > img.shape[1] - 1:
                boundaries.append((y, x))
                break

            elif img[yn, xn] != label:
                boundaries.append((y, x))
                break
    return boundaries


def draw_boundaries(img, label=1):
    bb = np.zeros_like(img)
    bb[img == label] = 1
    for y, x in get_boundaries(bb):
        bb[y, x] = 2
    return bb


def perimetr(img, label=1):
    return len(get_boundaries(img, label))


def circularity(img, label=1):
    return (perimetr(lb, label)**2)/area(img, label)


def distance(p1, p2):
    return ((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)**0.5


def radial_dist(img, label=1):
    r, c = centroid(img, label)
    bound = get_boundaries(img, label)
    K = len(bound)
    rd = 0
    for rk, ck in bound:
        rd += distance((r, c), (rk, ck))
    return rd / K


def std_radial(img, label=1):
    r, c = centroid(img, label)
    bound = get_boundaries(img, label)
    K = len(bound)
    rd = radial_dist(img, label)
    sr = 0
    for rk, ck, in bound:
        sr += (distance((r, c), (rk, ck)) - rd)**2
    return (sr/K)**0.5


def circularity_std(img, label=1):
    return radial_dist(img, label)/std_radial(img, label)


def moment_rc(lb, label=1):
    A = area(lb, label)
    r, c = centroid(lb, label)
    pos = np.where(lb == label)
    mrc = np.sum((pos[0]-r)*(pos[1]-c))
    return mrc / A



if __name__ == "__main__":


    lb = np.zeros((16, 16))
    lb[4:, :4] = 2
    lb[3:10, 8:] = 1
    lb[[3, 4, 3], [8, 8, 9]] = 0
    lb[[8, 9, 9], [8, 8, 9]] = 0

    lb[[3, 4, 3], [-2, -1, -1]] = 0
    lb[[9, 8, 9], [-2, -1, -1]] = 0

    lb[12:-1, 6:9] = 3
    plt.imshow(lb)
    plt.show()


    img = np.load('coins.npy.txt')

    lb = label(img)
    n = np.max(lb)
    ars = []
    for i in range(1, n+1):
        ars.append(azrea(lb, i))

    d = {}
    for ar in ars:
        if ar in d:
            d[ar] += 1
        else:
            d[ar] = 1

    nomin = {69: 1, 145: 2, 305: 5, 609: 10}
    total = 0
    for key in d:
        total += d[key]*nomin[key]
