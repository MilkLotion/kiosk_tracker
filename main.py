import cv2
import sys
import time

import EyeTracker as EyeTracker
import HandTracker
# import sendsocket

if __name__ == '__main__' :
    eyeT = EyeTracker.EyeT()
    handT = HandTracker.HandT()
    # send = sendsocket.sendS()

    cap = cv2.VideoCapture(0)
    # cap = cv2.VideoCapture('sample.mp4')
    # Frame
    pTime = 0
    cTime = 0

    while cap.isOpened() :
        ret, img = cap.read()

        # 이미지 처리
        img = cv2.flip(img, 1)
        img = cv2.resize(img, dsize=None, fx=0.7, fy=0.7)

        if not ret :
            break

        eyemsg = eyeT.main(img)
        handmsg = handT.main(img)

        sndmsg = str(eyemsg) + str(handmsg)

        # send.send(sndmsg)

        ## print fps
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_COMPLEX, 3, (0, 0, 255), 3)
        ##

        cv2.imshow('main', img)

        #tracker test waitKey
        inputKey = cv2.waitKey(1)
        if inputKey == 27:
            break
        elif inputKey == ord('r') or inputKey == ord('R'):
            print('Push Reset Button !!!')
            eyeT.Reset()
        elif inputKey == ord('x') or inputKey == ord('X'):
            print('pause')
            cv2.waitKey(0)

    cap.release()
    sys.exit('System OFF')