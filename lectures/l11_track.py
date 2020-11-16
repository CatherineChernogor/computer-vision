import cv2
import numpy as np

cam = cv2.VideoCapture(0)

cv2.namedWindow('camera', cv2.WINDOW_KEEPRATIO)
cv2.namedWindow('roi', cv2.WINDOW_KEEPRATIO)

roi = None

while cam.isOpened():
    ret, frame = cam.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break

    if key == ord('r'):
        r = cv2.selectROI('roi', gray)
        roi = gray[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]
        cv2.imshow('roi', roi)
        cv2.destroyWindow('roi seceltion')

    if roi is not None:
        res = cv2.matchTemplate(gray, roi, cv2.TM_CCORR_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        top_left = max_loc
        botton_right = (top_left[0]+roi.shape[1], top_left[1]+roi.shape[0])

        cv2.rectangle(frame, top_left, botton_right, 255, 2)
    cv2.imshow('camera', frame)

cam.release()
cv2.destroyAllWindows()
