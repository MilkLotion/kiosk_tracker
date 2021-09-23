import math
import time

import cv2

import HandDetectionModule as hdm

def judgmentDirection(dx, dy):
    radian = math.atan2(dy, dx)
    degree = int(math.degrees(radian))
    # print(degree)
    if degree < 45:
        return "left"
    elif degree > 125:
        return "right"
    else:
        return "center"


# for 프레임 속도
pTime = 0
cTime = 0

cap = cv2.VideoCapture(0)
detector = hdm.handDetector(trackingCon=0.7)

# 손가락의 끝
tipIds = [4, 8, 12, 16, 20]

while True:
    success, img = cap.read()
    # 손 찾기
    img = detector.findHands(img)
    # 지정한 mark 위치 찾기
    lmList = detector.findPosition(img, draw=False)

    if len(lmList) != 0:
        finger_direction = ""

        # hand mark(중지, 손목) 사이의 기울기로 방향을 판단한다
        x = lmList[12][1] - lmList[0][1]
        y = lmList[0][2] - lmList[12][2]
        finger_direction = judgmentDirection(x, y)

        cv2.putText(img, finger_direction, (20, 225), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

    # FPS
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, f'FPS: {int(fps)}', (400, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

    cv2.imshow("Image", img)
    cv2.waitKey(5)
