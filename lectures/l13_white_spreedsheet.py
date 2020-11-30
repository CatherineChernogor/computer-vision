# -*- coding: utf-8 -*-
"""
Created on Mon Nov 30 11:22:58 2020

@author: Stephanie
"""
import cv2

cv2.namedWindow("Camera", cv2.WINDOW_KEEPRATIO)
cam = cv2.VideoCapture(0)

while cam.isOpened():
    ret, frame = cam.read()
    
    
    cnts, hierarchy = cv2.findContours(frame, cv2.)
    print(len(cnts))
    cv2.imshow("Camera", frame)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()


