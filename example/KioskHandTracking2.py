import math
import time
from socket import *

import cv2

import HandDetectionModule as htm

# 소켓 통신
HOST = '127.0.0.1'
PORT = 10000

ADDR = (HOST, PORT)

clientSocket = socket(AF_INET, SOCK_STREAM)  # 서버에 접속하기 위한 소켓을 생성한다.
clientSocket.connect(ADDR)  # 서버에 접속을 시도한다.

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

class HandMotionDetector:
    def __init__(self):
        self.detector = htm.handDetector(trackingCon=0.7)

        # 손가락의 끝
        self.tipIds = [4, 8, 12, 16, 20]




def main():
    # for 프레임 속도
    pTime = 0
    cTime = 0

    cap = cv2.VideoCapture(0)
    detector = HandMotionDetector()

    # 보내냐 마느냐
    isSend = False
    count = 0
    premsg = ""
    while True:
        success, img = cap.read()
        # 손 찾기
        img = detector.findHands(img)
        # 지정한 mark 위치 찾기
        lmList = detector.findPosition(img, draw=False)
        # 메시지
        msg = ""
        # 클릭
        click = ""
        # 손 방향
        direction = ""
        # 손가락 갯수
        totalFingers = ""

        if len(lmList) != 0:
            fingers = []

            # 엄지는 접히는 방식이 달라 따로 처리한다. x축 이용
            if lmList[tipIds[0]][1] > lmList[tipIds[0] - 1][1]:
                fingers.append(1)
            else:
                fingers.append(0)

            # 검지부터 새끼손가락 까지
            for id in range(1, 5):
                # 손가락 끝(검지:8)이 손가락 두 번째 마디 (검지: 7)보다 위로 올라가면(y 좌표가 작아지면) 손가락을 폈다고 판단한다.
                if lmList[tipIds[id]][2] < lmList[tipIds[id] - 1][2]:
                    fingers.append(1)
                else:
                    fingers.append(0)

            # 손가락 개수 카운팅
            totalFingers = str(fingers.count(1))

            # 엄지와 검지의 좌표를 구한다
            x1, y1 = lmList[4][1], lmList[4][2]
            x2, y2 = lmList[12][1], lmList[12][2]
            # 엄지와 검지 끝 사이의 거리를 구한다.
            length = math.hypot(x2 - x1, y2 - y1)

            # 거리가 30이하면 클릭이라 판단
            if length < 30:
                click = "1"
            else:
                click = "0"

            # hand mark(중지, 손목) 사이의 기울기로 방향을 판단한다
            x = lmList[12][1] - lmList[0][1]
            y = lmList[0][2] - lmList[12][2]
            direction += judgmentDirection(x, y)

            msg = click + totalFingers + direction

            cv2.putText(img, msg, (20, 225), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

            if msg == premsg:
                count += 1

            if (isSend is True) and count >= 90:
                clientSocket.send(msg.encode())
                isSend = False
                count = 0

            premsg = msg

        if totalFingers == "0" and click == "0" and direction == "center":
            print("move")
            isSend = True

        # FPS
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

        cv2.imshow("Image", img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()
