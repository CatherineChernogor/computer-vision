import cv2
import numpy as np
import matplotlib.pyplot as plt


rose = cv2.imread('lectures/src/rose.jpg')

hsv = cv2.cvtColor(rose, cv2.COLOR_BGR2HSV)


lower_red = np.array([0, 200, 100])
upper_red = np.array([10, 255, 255])

mask = cv2.inRange(hsv, lower_red, upper_red)

cv2.namedWindow('original', cv2.WINDOW_KEEPRATIO)
cv2.imshow('original', mask)
cv2.waitKey(0)
cv2.destroyAllWindows()
