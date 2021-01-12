import pyautogui
import mss
import numpy as np
import cv2

monitor = {"top": 180, "left": 0, "width": 800, "height": 400}
kernel1 = np.ones((4,4), dtype=np.uint8)
kernel2 = np.ones((2,3), dtype=np.uint8)


def get_dist(x2, x1, y2, y1):
    return np.sqrt((x2-x1)**2+(y2-y1)**2)


def order_points(pts):
    result = np.zeros((4, 2), dtype="f4")

    s = pts.sum(axis=1)
    result[0] = pts[np.argmin(s)]  # top-left
    result[2] = pts[np.argmax(s)]  # bottom-right

    s = np.diff(pts, axis=1)
    result[1] = pts[np.argmin(s)]  # top-right
    result[3] = pts[np.argmax(s)]  # bottom-left

    return result


def start_game():
    pyautogui.click(x=+100, y=200)
    pyautogui.press('space')


def jump():
    pyautogui.press('space')


def check():
    with mss.mss() as sct:

        img = np.array(sct.grab(monitor))
        dino = img[50:350, 20:200]

        gray = cv2.cvtColor(dino, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)

        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel1)

        thresh = cv2.morphologyEx(thresh, cv2.MORPH_DILATE, kernel2)

        cnts, h = cv2.findContours(
            thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        # cv2.drawContours(dino, cnts, -1, (255, 255, 0), 2)

        objs = []

        for c in cnts[::-1][:3]:
            rect = cv2.minAreaRect(c)
            box = cv2.boxPoints(rect)
            dist = get_dist(box[0][0], box[1][0], box[0][1], box[1][1]) 
            print("dist", dist)
            if dist > 10 and dist < 150:
                objs.append(box)
                # objs.append(order_points(box))
                # cv2.drawContours(dino, [np.int0(box)], 0, (0, 0, 255), 2)

        print("amount", len(objs), '\n')

        # cv2.imshow("check", dino)
        # cv2.imshow("thresh", thresh)
        # if cv2.waitKey(0) == ord("q"):
        #     cv2.destroyAllWindows()

        if len(objs) == 1:
            return False
        else:
            return True

start_game()
while True:
    if check():
        jump()
