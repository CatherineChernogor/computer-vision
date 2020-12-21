import cv2
import numpy as np
import matplotlib.pyplot as plt
import math

flimit = 250
slimit = 255


def fupdate(value):
    global flimit
    flimit = value


def supdate(value):
    global slimit
    slimit = value


def get_dist(x1, y1, x2, y2):
    return math.sqrt((x2-x1)**2+(y2-y1)**2)


def get_sheet_points(x, y):
    points = []
    for _x, _y in zip(x, y):
        points.append([_x, _y])
    return np.array(points).reshape(4, 2)

def get_sheet_shape(points):
    p1 = points[0]
    p2 = points[1]
    p3 = points[2]
    p4 = points[3]

    widthA = np.sqrt(((p4[0] - p3[0]) ** 2) + ((p4[1] - p3[1]) ** 2))
    widthB = np.sqrt(((p2[0] - p1[0]) ** 2) + ((p2[1] - p1[1]) ** 2))
    # ...and now for the height of our new image
    heightA = np.sqrt(((p2[0] - p4[0]) ** 2) + ((p2[1] - p4[1]) ** 2))
    heightB = np.sqrt(((p1[0] - p3[0]) ** 2) + ((p1[1] - p3[1]) ** 2))
    # take the maximum of the width and height values to reach
    # our final dimensions
    maxWidth = max(int(widthA), int(widthB))
    maxHeight = max(int(heightA), int(heightB))

    return maxWidth, maxHeight

def order_points(pts):
    result = np.zeros((4, 2), dtype="f4")
    s = pts.sum(axis=1)
    result[0] = pts[np.argmin(s)]  # top-left
    result[2] = pts[np.argmax(s)]  # bottom-right
    s = np.diff(pts, axis=1)
    result[1] = pts[np.argmin(s)]  # top-right
    result[3] = pts[np.argmax(s)]  # bottom-left
    return result


cam = cv2.VideoCapture(0)
cv2.namedWindow("Camera", cv2.WINDOW_KEEPRATIO)
cv2.namedWindow("Mask", cv2.WINDOW_KEEPRATIO)
cv2.namedWindow("Sheet", cv2.WINDOW_KEEPRATIO)
cv2.namedWindow("Bin Sheet", cv2.WINDOW_KEEPRATIO)
cv2.namedWindow("New Sheet", cv2.WINDOW_KEEPRATIO)

cv2.createTrackbar('f', 'Mask', flimit, 255, fupdate)
cv2.createTrackbar('s', 'Mask', slimit, 255, supdate)

kernel = np.ones((7, 7))

while cam.isOpened():
    ret, frame = cam.read()
    converted = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)
    mask = cv2.inRange(converted, np.array(
        [0, flimit, 0]), np.array([150, slimit, 255]))

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.GaussianBlur(mask, (5, 5), 0)

    contours = cv2.findContours(
        mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
    if len(contours) > 0:
        paper = max(contours, key=cv2.contourArea)

        rect = cv2.minAreaRect(paper)
        box = cv2.boxPoints(rect)
        box_x = [int(i[0]) for i in box]
        box_y = [int(i[1]) for i in box]

        for p in box:
            cv2.circle(frame, tuple(p), 6, (0, 0, 255), 2)

        if box_x:
            sheet = frame[min(box_y): max(box_y), min(box_x): max(box_x)]
            # gray_sheet = cv2.cvtColor(sheet, cv2.COLOR_BGR2GRAY)

            #bin_sheet = mask[min(box_y): max(box_y), min(box_x): max(box_x)]
            new_sheet = sheet.copy()
            #new_sheet[bin_sheet == 0] = 0
            # new_sheet = cv2.cvtColor(sheet, cv2.COLOR_GRAY2BGR)
            if sheet.size > 0:
                cv2.imshow("Sheet", sheet)
                cv2.imshow("New Sheet", new_sheet)

                cv2.drawContours(frame, [np.int0(box)], 0, (0, 255, 0), 2)
                cv2.drawContours(frame, [paper], -1, (0, 0, 255), 2)

                pts = get_sheet_points(box_x, box_y)
                pts = order_points(pts)

                rows, cols = get_sheet_shape(pts)
                pts2 = np.float32([[0, 0], [0, rows-1],  [cols-1, rows-1], [cols-1, 0],])

                M = cv2.getPerspectiveTransform(pts, pts2)
                warp = cv2.warpPerspective(new_sheet, M, (cols, rows))
                cv2.imshow("Bin Sheet", warp)


    cv2.imshow("Camera", frame)
    cv2.imshow("Mask", mask)
    key = cv2.waitKey(1)
    if key == ord('p'):
        cv2.imwrite(
            "D:\_Progromouse\computer-vision\lectures\screen.png", frame)
    if key == ord('q'):
        p4eak

cam.release()
cv2.destroyAllWindows()
print(pts, pts2)

# print(sheet_points)
# print(new_sheet_points)


# def get_sheet_shape(points):
#     p1 = points[0]
#     p2 = points[1]
#     p3 = points[2]

#     x = int(get_dist(p1[0], p1[1], p2[0], p2[1]))
#     y = int(get_dist(p2[0], p3[1], p3[0], p2[1]))
#     return x, y


# def get_sheet_points(x, y):

#     p3 = [x[0], y[0]]

#     for _x, _y in zip(x, y):
#         if p3[0] < _x and p3[1] < _y:
#             p3 = [_x, _y]

#     p1 = [p3[0], p3[1]]
#     p2 = [p3[0], p3[1]]
#     p4 = [p3[0], p3[1]]
#     max_dist = 0
#     for _x, _y in zip(x, y):
#         if max_dist < get_dist(_x, _y, p3[0], p3[1]):
#             max_dist = get_dist(_x, _y, p3[0], p3[1])
#             p1 = [_x, _y]

#     for _x, _y in zip(x, y):
#         if (_x != p1[0] and _y != p1[1]):
#             if (_x != p3[0] and _y != p3[1]):

#                 if p1[0] < _x and p3[1] > _y:
#                     p2 = [_x, _y]

#                 if p3[0] > _x and p1[1] < _y:
#                     p4 = [_x, _y]

#     return [p1, p2, p3, p4]





# def get_new_sheet_points(rows, cols, points):
#     min_dist = get_dist(points[0][0], points[0][1], 0, 0)
#     for (i, p) in enumerate(points):
#         dist = get_dist(p[0], p[1], 0, 0)
#         if min_dist > dist:
#             min_dist = dist
#             closest_i = i

#     print(closest_i, points)

#     new_points = [0, 0, 0, 0]
#     def_points = []
#     next_p = 0
#     if (closest_i + 1 == len(new_points)):
#         next_p = points[0]
#     else:
#         next_p = points[closest_i+1]

#     if next_p[0] < next_p[1]:
#         # rows first
#         def_points = [[0, 0], [0, rows], [cols, 0], [cols, rows]]
#     else:
#         # cols first
#         def_points = [[0, 0], [cols, 0], [0, rows], [cols, rows]]

#     j = closest_i
#     for i in range(len(new_points)):

#         new_points[j] = def_points[i]
#         if (j + 1 == len(new_points)):
#             j = 0
#         else:
#             j += 1

#     return new_points

