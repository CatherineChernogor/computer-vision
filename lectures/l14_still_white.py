import cv2
import numpy as np

cv2.namedWindow("Camera", cv2.WINDOW_KEEPRATIO)
cv2.namedWindow("Background", cv2.WINDOW_KEEPRATIO)
cv2.namedWindow("Delta", cv2.WINDOW_KEEPRATIO)

cam = cv2.VideoCapture(0)

bg = None
fc = 0
while cam.isOpened():
    ret, frame = cam.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (15, 15), 0)


    if fc < 30:
        if bg is None:
            bg = gray.copy().astype("f4")
            cv2.accumulateWeighted(gray, bg, 0.05)
    else:
        bg =  bg.astype("uint8")
        cv2.imshow("Background", bg)
        delta = cv2.absdiff(bg, gray)
        binary = cv2.threshold(delta, 100, 255, cv2.THRESH_BINARY)[1]
        cnts, h = cv2.findContours(
        binary.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        cv2.drawContours(frame, cnts, -1, (0, 0, 255), 1)
        cv2.imshow("Delta", binary)

    cv2.imshow("Camera", frame)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

    if key == ord('p'):
        if binary is not None:
            cv2.imwrite("D:\_Progromouse\computer-vision\lectures\screen.png", np.hstack([frame[:, :, 0], gray, binary]))
           
       
    fc+= 1

cam.release()
cv2.destroyAllWindows()
