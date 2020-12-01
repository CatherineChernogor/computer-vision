import cv2
import numpy as np


cv2.namedWindow("Camera", cv2.WINDOW_KEEPRATIO)
cv2.namedWindow("thresh", cv2.WINDOW_KEEPRATIO)
cv2.namedWindow("sheet", cv2.WINDOW_KEEPRATIO)

cv2.namedWindow("settings", cv2.WINDOW_KEEPRATIO)

cam = cv2.VideoCapture(0)


def nothing(*arg):
    pass


def get_rect(cnts):
    max_area = 0
    for cnt in cnts:
        rect = cv2.minAreaRect(cnt)
        area = rect[1][0]*rect[1][1]

        if max_area < area:
            max_area = area
            box = cv2.boxPoints(rect)

    return box, max_area


cv2.createTrackbar('h1', 'settings', 0, 255, nothing)
cv2.createTrackbar('s1', 'settings', 0, 255, nothing)
cv2.createTrackbar('v1', 'settings', 0, 255, nothing)
cv2.createTrackbar('h2', 'settings', 255, 255, nothing)
cv2.createTrackbar('s2', 'settings', 255, 255, nothing)
cv2.createTrackbar('v2', 'settings', 255, 255, nothing)
crange = [0, 0, 0, 0, 0, 0]


while cam.isOpened():
    ret, frame = cam.read()

    blurred = cv2.GaussianBlur(frame, (9, 9), 0)
    mask = cv2.erode(blurred, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    hsv = cv2.cvtColor(mask, cv2.COLOR_BGR2HSV)

    # считываем значения бегунков
    # h1 = cv2.getTrackbarPos('h1', 'settings')
    # s1 = cv2.getTrackbarPos('s1', 'settings')
    # v1 = cv2.getTrackbarPos('v1', 'settings')
    # h2 = cv2.getTrackbarPos('h2', 'settings')
    # s2 = cv2.getTrackbarPos('s2', 'settings')
    # v2 = cv2.getTrackbarPos('v2', 'settings')

    # формируем начальный и конечный цвет фильтра
    # h_min = np.array((h1, s1, v1), np.uint8)
    # h_max = np.array((h2, s2, v2), np.uint8)

    h_min = np.array((80, 18, 170), np.uint8)
    h_max = np.array((125, 35, 255), np.uint8)

    thresh = cv2.inRange(hsv, h_min, h_max)

    cnts, h = cv2.findContours(
        thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    cv2.drawContours(frame, cnts, -1, (255, 0, 0), 1)

    # наш лист это самый большой прямоугольник
    box, max_area = get_rect(cnts)

    cv2.drawContours(frame, [np.int0(box)], 0, (0, 255, 0), 2)

    box_x = [int(i[0]) for i in box]
    box_y = [int(i[1]) for i in box]

    cv2.circle(frame, (box_x[0], box_y[0]), 5, (255, 255, 127), 2)
    cv2.circle(frame, (box_x[2], box_y[1]), 10, (255, 255, 127), 2)
    cv2.circle(frame, (box_x[2], box_y[2]), 15, (255, 255, 127), 2)
    cv2.circle(frame, (box_x[3], box_y[3]), 20, (255, 255, 127), 2)

    print(frame.shape)

    if max_area > 20:
        sheet = frame[min(box_y): max(box_y)][min(box_x): max(box_x)]
        
        cv2.imshow("sheet", sheet)

    # try to get cnts
    # x,y,w,h = cv2.boundingRect(cntr)    


    
    # rows, cols, _ = sheet.shape

    # pts1 = np.float32(sheet_coord)
    # pts2 = np.float32()

    # M = cv2.getPerspectiveTransform(pts1, pts2)
    # aff_img = cv2.warpPerspective(sheet, M, (cols, rows))

    cv2.imshow("Camera", frame)
    cv2.imshow("thresh", thresh)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()

print(len(cnts))
print(box)
print(box_x)
print(frame.shape)
print(min(box_x), max(box_x), min(box_y), max(box_y))
