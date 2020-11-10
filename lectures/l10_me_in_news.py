import cv2
import numpy as np

cam = cv2.VideoCapture(0)

if not cam.isOpened():
    raise RuntimeError("camera does not work")

ret, subing = cam.read()

news = cv2.imread('lectures/src/news.jpg')

rows, cols, _ = subing.shape

pts1 = np.float32([[0, 0], [0, rows], [cols, 0], [cols, rows]])
pts2 = np.float32([[17, 25], [41, 292], [434, 55], [434, 268]])

M = cv2.getPerspectiveTransform(pts1, pts2)
aff_img = cv2.warpPerspective(subing, M, (cols, rows))[:300, :500]
mask = np.zeros((aff_img.shape[0], aff_img.shape[1]), dtype='uint8')

for y in range(aff_img.shape[0]):
    for x in range(aff_img.shape[1]):
        if np.any(aff_img[y, x, :] != 0):
            mask[y, x] = 255

mask_inv = ~mask
roi = news[:aff_img.shape[0], :aff_img.shape[1]]
bg = cv2.bitwise_and(roi, roi, mask=mask_inv)


while True:
    ret, subing = cam.read()
    aff_img = cv2.warpPerspective(subing, M, (cols, rows))[:300, :500]
    fg = cv2.bitwise_and(aff_img, aff_img, mask=mask)

    combined = cv2.add(bg, fg)
    news[:combined.shape[0], :combined.shape[1]] = combined
    cv2.imshow('video', news)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break


cam.release()
cv2.destroyAllWindows()
