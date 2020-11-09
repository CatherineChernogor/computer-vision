import cv2
import numpy as np
import matplotlib.pyplot as plt


mush = cv2.imread('lectures/src/mushroom.jpg')
logo = cv2.imread('lectures/src/cvlogo.png')
logo = cv2.resize(logo, (logo.shape[0]//2, logo.shape[1]//2))

print(logo.shape)


roi = mush[:logo.shape[0], :logo.shape[1]]

logo_gray = cv2.cvtColor(logo, cv2.COLOR_BGR2GRAY)
ret, mask = cv2.threshold(logo_gray, 10, 255, cv2.THRESH_BINARY)

mask_inv = cv2.bitwise_not(mask)

img1_bg = cv2.bitwise_and(roi, roi, mask_inv)
img2_bg = cv2.bitwise_and(logo, logo, mask_inv)

combined = cv2.add(img1_bg, img2_bg)

mush[:logo.shape[0], :logo.shape[1]] = combined
cv2.namedWindow('original', cv2.WINDOW_KEEPRATIO)
cv2.imshow('original', mush)
cv2.waitKey(0)
cv2.destroyAllWindows()
