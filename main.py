import math

import cv2
# import cvzone
import numpy as np
from HandTrackingModule import *
# from cvzone.HandTrackingModule import HandDetector

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 960)
detector = HandDetector(detectionCon=0.8, maxHands=2)
x = [300, 245, 200, 170, 145, 130, 112, 103, 93, 87, 80, 75, 70, 67, 62, 59, 57]
y = [20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]
coff = np.polyfit(x, y, 2)  # y = Ax^2 + Bx + C

while True:
    success, img = cap.read()
    # print('size = ', img.shape)
    hands = detector.findHands(img, draw=False)
    if hands:
        lmList = hands[0]['lmList']
        x, y, w, h = hands[0]['bbox']
        x1, y1, x3 = lmList[5]
        x2, y2, x3 = lmList[17]

        distance = int(math.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2))
        A, B, C = coff
        distanceCM = A * distance ** 2 + B * distance + C

        print(distanceCM, distance)

        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(img, f'{int(distanceCM)} cm', (x + 5, y - 10),cv2.FONT_HERSHEY_SIMPLEX,1,(255, 0, 245),3)
        # #各参数依次是：图片，添加的文字，左上角坐标，字体，字体大小，颜色，字体粗细
    cv2.imshow("video", img)
    # cv2.waitKey(1)
    if (cv2.waitKey(1) & 0xFF == ord('q')) or (cv2.waitKey(1) & 0xFF == ord('Q')):
        break
cap.release()  # 关闭视频读取
cv2.destroyAllWindows()  # 关闭所有windows窗口
