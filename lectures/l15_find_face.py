import cv2
import numpy as np
import matplotlib.pyplot as plt


cv2.namedWindow("Camera", cv2.WINDOW_KEEPRATIO)
cam = cv2.VideoCapture(0)

if not cam.isOpened():
    raise RuntimeError("Camera broken")

cascade = cv2.CascadeClassifier(
    'lectures\src\haarcascade_frontalface_default.xml')

while cam.isOpened():
    ret, frame = cam.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

    cv2.imshow("Camera", frame)
    key = cv2.waitKey(1)
    if key == ord('p'):
        cv2.imwrite(
            "D:\_Progromouse\computer-vision\lectures\screen.png", frame)
    if key == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
