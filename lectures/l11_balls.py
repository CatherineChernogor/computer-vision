import cv2
import numpy as np

position = [0, 0]
bgr_color = []
hsv_color = []


def on_mouse_click(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        global position
        position = [y, x]
        print(position)


cam = cv2.VideoCapture(0)
cv2.namedWindow('camera', cv2.WINDOW_KEEPRATIO)
cv2.setMouseCallback('camera', on_mouse_click)

measures = []
while cam.isOpened():
    ret, frame = cam.read()

    pxl_color = frame[position[0], position[1], :]
    measures.append(pxl_color)
    if len(measures) >= 10:
        bgr_color = np.uint8([[np.average(measures, 0)]])
        hsv_color = cv2.cvtColor(bgr_color, cv2.COLOR_BGR2HSV)
        bgr_color = bgr_color[0][0]
        hsv_color = hsv_color[0][0]
        measures.clear()

    cv2.putText(frame, f"BGR = {bgr_color}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 127))
    cv2.putText(frame, f"HSV = {hsv_color}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 127))
    cv2.circle(frame, (position[1], position[0]), 5, (0, 255, 127))
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    cv2.imshow('camera', frame)


cam.release()
cv2.destroyAllWindows()
