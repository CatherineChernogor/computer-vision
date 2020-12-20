import cv2
import numpy as np
import matplotlib.pyplot as plt


cv2.namedWindow("Camera", cv2.WINDOW_KEEPRATIO)
cam = cv2.VideoCapture(0)

if not cam.isOpened():
    raise RuntimeError("Camera broken")

modelFile = 'lectures/src/res10_300x300_ssd_iter_140000_fp16.caffemodel'
configFile = 'lectures/src/deploy.proto'
net = cv2.dnn.readNetFromCaffe(configFile, modelFile)


def censore(image):
    cube = 20
    for y in range(0, image.shape[0], cube):
        for x in range(0, image.shape[1], cube):
            new_value = np.mean(image[y: y+cube, x:x+cube, :])
            image[y: y+cube, x:x+cube, :] = new_value
    return image

while cam.isOpened():
    ret, frame = cam.read()

    blob = cv2.dnn.blobFromImage(frame, 1.0, (299, 299), [
                                 104, 117, 123], False, False)
    net.setInput(blob)
    detections = net.forward()

    boxes = []
    for i in range(detections.shape[2]):
        params = detections[0, 0, i, :]
        confidece = params[2]
        if confidece > 0.8:
            x1 = int(params[3] * frame.shape[1])
            y1 = int(params[4] * frame.shape[0])
            x2 = int(params[5] * frame.shape[1])
            y2 = int(params[6] * frame.shape[0])

            w = x2-x1
            h = y2-y1
            
            censored = frame
            censored[y1:y1+h, x1:x1+w] = censore(frame[y1:y1+h, x1:x1+w])
            # cv2.rectangle(frame, (x1, y1), (x1+w, y1+h), (255, 0, 0), 2)

    cv2.imshow("Camera", censored)
    key = cv2.waitKey(1)
    if key == ord('p'):
        cv2.imwrite(
            "D:\_Progromouse\computer-vision\lectures\screen.png", frame)
    if key == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
