import cv2
import numpy as np
import matplotlib.pyplot as plt


news = cv2.imread('lectures/src/news.jpg')
chebu = cv2.imread('lectures/src/cheburashka.jpg')

rows, cols, _ = chebu.shape

pts1 = np.float32([[0,0],[0, rows],[cols, 0],[cols, rows]])
pts2 = np.float32([[17,25],[41, 292],[434, 55],[434, 268]])

# cv2.namedWindow('original', cv2.WINDOW_KEEPRATIO)
# cv2.imshow('original', chebu)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
x = [p[0] for p in pts2]
y = [p[1] for p in pts2]

plt.scatter(x, y)

plt.imshow(news)
plt.show()