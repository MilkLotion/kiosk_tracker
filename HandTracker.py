import cv2
import math

import HandDetectionModule as htm

def judgmentDirection(dx, dy):
    radian = math.atan2(dy, dx)
    degree = int(math.degrees(radian))
    # print(degree)
    if degree < 45:
        return 1
    elif degree > 125:
        return -1
    else:
        return 0


class HandMotionDetector:
    def __init__(self):
        # 손가락의 끝
        self.tipIds = [4, 8, 12, 16, 20]

        self.count = None
        self.isClick = None
        self.direction = None

    def fingerCount(self, lmList):
        fingers = []

        # 엄지는 접히는 방식이 달라 따로 처리한다. x축 이용
        if lmList[self.tipIds[0]][1] > lmList[self.tipIds[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)

        # 검지부터 새끼손가락 까지
        for id in range(1, 5):
            # 손가락 끝(검지:8)이 손가락 두 번째 마디 (검지: 7)보다 위로 올라가면(y 좌표가 작아지면) 손가락을 폈다고 판단한다.
            if lmList[self.tipIds[id]][2] < lmList[self.tipIds[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)

        # 손가락 개수 카운팅
        self.count = fingers.count(1)

    def click(self, lmList):
        # 엄지와 검지의 좌표를 구한다
        x1, y1 = lmList[4][1], lmList[4][2]
        x2, y2 = lmList[12][1], lmList[12][2]
        # 엄지와 검지 끝 사이의 거리를 구한다.
        length = math.hypot(x2 - x1, y2 - y1)

        # 거리가 30이하면 클릭이라 판단
        if length < 30:
            self.isClick = 1
        else:
            self.isClick = 0

    def handDirection(self, lmList):
        # hand mark(중지, 손목) 사이의 기울기로 방향을 판단한다
        x = lmList[12][1] - lmList[0][1]
        y = lmList[0][2] - lmList[12][2]
        self.direction = judgmentDirection(x, y)
#########


tracker = htm.handDetector()
detector = HandMotionDetector()


class HandT() :
    def main(self, img):
        # 손 찾기
        img = tracker.findHands(img)
        # 지정한 mark 위치 찾기
        lmList = tracker.findPosition(img, draw=False)
        # 메시지
        msg = ""

        # tracker mark is not none
        if len(lmList) != 0:
            # 손가락 개수 세기
            detector.fingerCount(lmList)
            # 클릭 판단하기
            detector.click(lmList)
            # 방향 판단하기
            detector.handDirection(lmList)

            # 메시지
            msg = str(detector.count) + str(detector.isClick) + str(detector.direction)
            cv2.putText(img, msg, (20, 225), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

            ###
        return msg