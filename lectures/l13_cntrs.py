# -*- coding: utf-8 -*-
"""
Created on Mon Nov 30 10:14:39 2020

@author: Stephanie
"""
import cv2

image = cv2.imread('defects.png')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

cnts, hierarchy = cv2.findContours(gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

arrow = cnts[0]
cv2.drawContours(image, cnts, 0, (255, 0, 0), 3)

print("Area = ", cv2.contourArea(arrow))
print("Perimeter = ", cv2.arcLength(arrow, False))

moments = cv2.moments(arrow)
print("Moments = ", moments)
centroid = (int(moments['m10']/moments['m00']), 
            int(moments['m01']/moments['m00']))

print("Centroid = ", centroid)

cv2.circle(image, centroid, 4, (0, 255, 0), 2)

eps = 0.001 * cv2.arcLength(arrow, True)
approx = cv2.approxPolyDP(arrow, eps, True)

for p in approx:
    cv2.circle(image, tuple(*p), 6, (0, 255, 0), 2)


hull = cv2.convexHull(arrow)
for i in range(1, len(hull)):
    cv2.line(image, tuple(*hull[i-1]), tuple(*hull[i]), (0, 255, 0), 2)
cv2.line(image, tuple(*hull[-1]), tuple(*hull[0]), (0, 255, 0), 2)


x, y, w, h = cv2.boundingRect(arrow)
cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

rect = cv2.minAreaRect(arrow)
box = cv2.boxPoints(rect)
import numpy as np
cv2.drawContours(image, [np.int0(box)], 0, (0, 255, 0), 2)


(x, y), radius = cv2.minEnclosingCircle(arrow)
center = int(x), int(y)
radius = int(radius)
cv2.circle(image, center, radius, (0, 255, 0), 2)

vx, vy, x, y = cv2.fitLine(arrow, cv2.DIST_L2, 0, 0.01, 0.01)
left_y=int(((-x * vy/ vx)+y))
right_y = int(((image.shape[0]-x)*vy/vx)+y)
cv2.line(image, (image.shape[0]-1, right_y), (0, left_y), (0, 255, 0), 2)

ellipse = cv2.fitEllipse(arrow)
cv2.ellipse(image, ellipse, (0, 255, 0), 2)




# for cnt in cnts:
#     print(len(cnts))
#     for pnt in cnt:
#         cv2.circle(image, tuple(*pnt), 5, (255,0,255), 1)

cv2.drawContours(image, cnts, -1, (255, 0, 0), 3)

cv2.namedWindow('image', cv2.WINDOW_KEEPRATIO)
cv2.imshow('image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()