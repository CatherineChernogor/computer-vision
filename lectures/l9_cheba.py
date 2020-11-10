import cv2
import numpy as np
import matplotlib.pyplot as plt


news = cv2.imread('lectures/src/news.jpg')
chebu = cv2.imread('lectures/src/cheburashka.jpg')

rows, cols, _ = chebu.shape

pts1 = np.float32([[0,0],[0, rows],[cols, 0],[cols, rows]])
pts2 = np.float32([[17,25],[41, 292],[434, 55],[434, 268]])

M = cv2.getPerspectiveTransform(pts1, pts2)
aff_img = cv2.warpPerspective(chebu, M, (cols, rows))


for y in range(aff_img.shape[0]):
    for x in range(aff_img.shape[1]):
        if np.any(aff_img[y, x, :]!= 0):
            news[y, x, :] = aff_img[y, x, :]

cv2.namedWindow('original', cv2.WINDOW_KEEPRATIO)
cv2.imshow('original', news)
cv2.waitKey(0)
cv2.destroyAllWindows()
# x = pts2[:, 0]
# y = pts2[:, 1]

# plt.scatter(x, y)

# plt.imshow(news)
plt.show()