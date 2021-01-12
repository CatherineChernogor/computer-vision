import cv2
import numpy as np

def get_dist(x2, x1, y2, y1):
    return np.sqrt((x2-x1)**2+(y2-y1)**2)    

def order_points(pts):
    result = np.zeros((4, 2), dtype="f4")

    s = pts.sum(axis=1)
    result[0] = pts[np.argmin(s)]  # top-left
    result[2] = pts[np.argmax(s)]  # bottom-right

    s = np.diff(pts, axis=1)
    result[1] = pts[np.argmin(s)]  # top-right
    result[3] = pts[np.argmax(s)]  # bottom-left

    return result

image = cv2.imread('t-rex/ex.png')

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
_, thresh = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)

cnts, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cv2.imshow('a', thresh)

cv2.drawContours(image, cnts[::-1][:3], -1, (255, 255, 0), 3)

objs = []

for c in  cnts[::-1][:3]:
    # вписываем прямоугольник
    rect = cv2.minAreaRect(c)
    box = cv2.boxPoints(rect)
    print(get_dist(box[0][0], box[1][0], box[0][1], box[1][1]))
    if get_dist(box[0][0], box[1][0], box[0][1], box[1][1]) > 3:

        objs.append(order_points(box))
    cv2.drawContours(image, [np.int0(box)], 0, (0, 255, 0), 2)

print(objs)
cv2.imshow('image', image)
cv2.waitKey()
cv2.destroyAllWindows()