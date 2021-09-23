import math
import time

import cv2

import HandDetectionModule as htm

# for 프레임 속도
pTime = 0
cTime = 0

cap = cv2.VideoCapture(0)
detector = htm.handDetector(trackingCon=0.7)

while True:
    success, img = cap.read()
    # 손 찾기
    img = detector.findHands(img)
    # 지정한 mark 위치 찾기
    lmList = detector.findPosition(img, draw=False)

    if len(lmList) != 0:
        # print(lmList[4], lmList[8])
        # 엄지와 검지의 좌표를 구한다
        x1, y1 = lmList[4][1], lmList[4][2]
        x2, y2 = lmList[8][1], lmList[8][2]

        # 엄지와 검지 끝 사이의 거리를 구한다.
        length = math.hypot(x2 - x1, y2 - y1)
        # print(length)

        # 거리가 30이하면 클릭이라 판단
        if length < 30:
            cv2.putText(img, "Click", (20, 220), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

    # FPS
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

    cv2.imshow("Image", img)
    cv2.waitKey(1)
