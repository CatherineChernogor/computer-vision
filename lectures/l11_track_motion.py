import cv2
import numpy as np

cam = cv2.VideoCapture(0)

cv2.namedWindow('camera', cv2.WINDOW_KEEPRATIO)
cv2.namedWindow('background', cv2.WINDOW_KEEPRATIO)

background = None

while cam.isOpened():
    ret, frame = cam.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

    if key == ord('b'):
        background = gray.copy()

    if background is not None:
        cv2.imshow('background', background)
        delta = cv2.absdiff(background, gray)
        thresh = cv2.threshold(delta, 25, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.dilate(thresh, None, iterations=2)
        cnts, _ = cv2.findContours(
            thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in cnts:
            area = cv2.contourArea(cnt)
            if area > 450:
                (x, y, w, h) = cv2.boundingRect(cnt)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    cv2.imshow('camera', frame)

cam.release()
cv2.destroyAllWindows()
