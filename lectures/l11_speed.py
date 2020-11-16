import cv2
import numpy as np


def set_upper(x):
    global colorUpper
    colorUpper[0] = x


def set_lower(x):
    global colorLower
    colorLower[0] = x


cam = cv2.VideoCapture(0)
cv2.namedWindow('camera', cv2.WINDOW_KEEPRATIO)
cv2.namedWindow('mask', cv2.WINDOW_KEEPRATIO)

cv2.createTrackbar('u', 'mask', 0, 255, set_upper)
cv2.createTrackbar('l', 'mask', 0, 255, set_lower)

colorLower = np.array([0, 0, 0], dtype='uint8')
colorUpper = np.array([255, 255, 255], dtype='uint8')
  
while cam.isOpened():
    ret, frame = cam.read()

    blurred = cv2.GaussianBlur(frame, (11, 11), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv, colorLower, colorUpper)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    cnts, _ = cv2.findContours(
        mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(cnts) > 0:
        c = max(cnts, key=cv2.contourArea)
        (curr_x, curr_y), radii = cv2.minEnclosingCircle(c)
        if radii > 10:
            cv2.circle(frame, (int(curr_x), int(curr_y)), int(radii), (0, 255, 255), 2)


    
    
    cv2.imshow('mask', mask)
    cv2.imshow('camera', frame)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
