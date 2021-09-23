import time

import cv2

import HandDetectionModule as htm

# for 프레임 속도
pTime = 0
cTime = 0

cap = cv2.VideoCapture(0)
detector = htm.handDetector(trackingCon=0.7)

# 손가락의 끝
tipIds = [4, 8, 12, 16, 20]

while True:
    success, img = cap.read()
    # 손 찾기
    img = detector.findHands(img)
    # 지정한 mark 위치 찾기
    lmList = detector.findPosition(img, draw=False)

    if len(lmList) != 0:
        fingers = []

        # 엄지는 접히는 방식이 달라 따로 처리한다. x축 이용
        if lmList[tipIds[0]][1] > lmList[tipIds[0] - 1][1]:
            fingers.append(1)
            # print("Index finger open")
        else:
            fingers.append(0)

        # 검지부터 새끼손가락 까지
        for id in range(1, 5):
            # 손가락 끝(검지:8)이 손가락 두 번째 마디 (검지: 7)보다 위로 올라가면(y 좌표가 작아지면) 손가락을 폈다고 판단한다.
            if lmList[tipIds[id]][2] < lmList[tipIds[id] - 1][2]:
                fingers.append(1)
                # print("Index finger open")
            else:
                fingers.append(0)

        # 손가락 개수 카운팅
        totalFingers = fingers.count(1)

        cv2.putText(img, str(totalFingers), (20, 60), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

    # FPS
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, f'FPS: {int(fps)}', (400, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

    cv2.imshow("Image", img)
    cv2.waitKey(5)
