import cv2
import numpy as np


def set_upper(x):
    global colorUpper
    colorUpper[0] = x


def set_lower(x):
    global colorLower
    colorLower[0] = x


cam = cv2.VideoCapture(0)
cv2.namedWindow('camera', cv2.WINDOW_KEEPRATIO)
cv2.namedWindow('mask', cv2.WINDOW_KEEPRATIO)

colorLower = np.array([0, 0, 0], dtype='uint8')
colorUpper = np.array([255, 255, 255], dtype='uint8')


def draw(cnts, frame):
    if len(cnts) > 0:
        c = max(cnts, key=cv2.contourArea)
        (curr_x, curr_y), radii = cv2.minEnclosingCircle(c)
        if radii > 10:
            cv2.circle(frame, (int(curr_x), int(curr_y)),
                       int(radii), (0, 255, 255), 2)


def get_cnts(hsv, colorLower, colorUpper):
    mask = cv2.inRange(hsv, colorLower, colorUpper)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)
    # cv2.imshow('mask', mask)

    cnts, _ = cv2.findContours(
        mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    return cnts


colors = {
    'blue': [[95, 100, 100], [110, 255, 255]],
    'green': [[50, 0, 100], [90, 255, 255]],
    'orange': [[10, 100, 100], [20, 255, 255]],

}


def get_coords(cnts):

    
    coords_x = []
    if (len(cnts) > 0):

        for cnt in cnts:
            for c in cnt:
                return c[0][0]
    else:
        return 0


while cam.isOpened():
    ret, frame = cam.read()

    blurred = cv2.GaussianBlur(frame, (11, 11), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    cnts1 = get_cnts(hsv, np.array(
        colors['blue'][0]), np.array(colors['blue'][1]))
    draw(cnts1, frame)
    cnts2 = get_cnts(hsv, np.array(
        colors['green'][0]), np.array(colors['green'][1]))
    draw(cnts2, frame)
    cnts3 = get_cnts(hsv, np.array(
        colors['orange'][0]), np.array(colors['orange'][1]))
    draw(cnts3, frame)


    ball_set = [
        ('blue', get_coords(cnts1)),
        ('green', get_coords(cnts2)),
        ('orange', get_coords(cnts3)),
    ]



    ball_set.sort(key=lambda i: i[1])
    for i, ball in enumerate(ball_set):
        if ball[1]!=0:
            cv2.putText(frame, f"{i}: {ball[0]}", (10, 30+20*i), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255))


    cv2.imshow('camera', frame)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break
cam.release()
cv2.destroyAllWindows()
