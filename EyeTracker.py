import copy

import cv2
import numpy as np

# eye roi thresh -> divide vertical half -> left, right eye roi
# make big cnt for only eye -> only eye roi erosion -> center is eye

threshold_file = open('threshold.txt', 'r')
threshold_val = int(threshold_file.readline())

class EyeT() :
    def __init__(self) :
        # search face model
        self.cascade_filename = 'haarcascade_frontalface_alt.xml'
        # 모델 불러오기
        self.cascade = cv2.CascadeClassifier(self.cascade_filename)

        # tracker setting
        # 트랙커 객체 생성자 함수 리스트 ---①
        self.tracker = None
        self.eyetracker = None
        self.trackerAlg = cv2.legacy.TrackerMOSSE_create

        self.isFirst = True
        self.faceCount = 0
        self.failCnt = 0

        self.LIrisFirst = True
        self.LIrisbox = None

        self.RIrisFirst = True
        self.RIrisbox = None

        # 최초 아이트래커 보정
        self.isSet = False
        self.setCnt = 0
        self.irisCC = [0, 0]
        self.irisRD = [0, 0]
        self.RDList = np.empty((0,2))
        # cc to rd size roi
        self.detectROI = None
        self.detectLR = "l"
        self.detectUD = "u"
        self.upsig = False

    def SearchFace(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # 얼굴 탐색
        face = self.cascade.detectMultiScale(gray,  # 입력 이미지
                                                scaleFactor=1.1,  # 이미지 피라미드 스케일 factor
                                                minNeighbors=5,  # 인접 객체 최소 거리 픽셀
                                                minSize=(20, 20)  # 탐지 객체 최소 크기
                                                )
        box = []
        # 얼굴 탐색 실패시 초기화
        if len(face) == 0:
            self.Reset()
            self.isFirst = True
            return box
        else:
            for box in face :
                return box

    # 얼굴 영역내의 눈 영역
    def SearchEye(self, faceROI):
        # search eyes area model
        net = cv2.dnn.readNet("YOLOv3-wd_final.weights", "YOLOv3-wd.cfg")
        classes = []
        with open("obj.names", "r") as f:
            classes = [line.strip() for line in f.readlines()]
        layer_names = net.getLayerNames()
        output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

        height, width, channel = faceROI.shape

        # Detecting objects locations(outs)
        blob = cv2.dnn.blobFromImage(faceROI, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outs = net.forward(output_layers)

        # Showing informations on the screen
        class_ids = []
        confidences = []
        boxes = []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    # Object detected
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    # Rectangle coordinates
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        # 노이즈제거 => 같은 물체에 대한 박스가 많은것을 제거(Non maximum suppresion)
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        '''
            Box : 감지된 개체를 둘러싼 사각형의 좌표
            Label : 감지된 물체의 이름
            Confidence : 0에서 1까지의 탐지에 대한 신뢰도
        '''
        for i in range(len(boxes)):
            if i in indexes:
                label = str(classes[class_ids[i]])
                if label == 'eyes':
                    boxes = boxes[0]

        return boxes

    # 눈 영역 내에서 눈동자 영역 추출
    # draw Contours
    def find_if_close(self, cnt1, cnt2):
        row1, row2 = cnt1.shape[0], cnt2.shape[0]
        for i in range(row1):
            for j in range(row2):
                dist = np.linalg.norm(cnt1[i] - cnt2[j])
                if abs(dist) < 3:
                    return True
                elif i == row1 - 1 and j == row2 - 1:
                    return False

    def bigcnt(self, contours):
        LENGTH = len(contours)
        status = np.zeros((LENGTH, 1))

        for i, cnt1 in enumerate(contours):
            x = i
            if i != LENGTH - 1:
                for j, cnt2 in enumerate(contours[i + 1:]):
                    x = x + 1
                    dist = self.find_if_close(cnt1, cnt2)
                    if dist:
                        val = min(status[i], status[x])
                        status[x] = status[i] = val
                    else:
                        if status[x] == status[i]:
                            status[x] = i + 1

        unified = []
        if len(status) == 0 :
            pass
        else :
            maximum = int(status.max()) + 1
            for i in range(maximum):
                pos = np.where(status == i)[0]
                if pos.size != 0:
                    cont = np.vstack([contours[i] for i in pos])
                    hull = cv2.convexHull(cont)
                    unified.append(hull)

        return unified

    def get_eye_shape(self, thresh) :
        # get eye width, height
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        val = [0, 0, 0, 0]
        # get eye rect
        unified = self.bigcnt(contours)

        size_array = []

        if len(unified) > 1 :
            for i in range(len(unified)) :
                size = cv2.contourArea(unified[i])
                size_array.append([size, unified[i]])
                # size_array.append(size)

            size_array.sort(key=lambda r: r[0], reverse=True)

        else :
            return val

        val = cv2.boundingRect(size_array[0][1])

        return val

    def getCenter(self, roi) :
        contours, hierarchy = cv2.findContours(roi, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        cnt_area = []
        for cnt in contours:
            # 중심점 좌표 구하기
            M = cv2.moments(cnt)

            cx = M['m10'] / (M['m00'] + 1e-5)
            cy = M['m01'] / (M['m00'] + 1e-5)

            cnt_area.append([cv2.contourArea(cnt), cx, cy])

        # 구한 중심점들 오름차순
        if cnt_area:
            cnt_area.sort(key=lambda r: r[0], reverse=True)

            # cv2.circle(thresh, self.iris_cc, 4, 0, -1)

            # cnt_pos = [cnt_area[0][1] / self.box_w, cnt_area[0][2] / self.box_h]
            #
            # height, width = thresh.shape[:2]
            # zeros = np.zeros((height, width, 3), np.uint8)
            # cv2.circle(zeros, (int(cnt_area[0][1]), int(cnt_area[0][2])), 2, (255, 0, 0), -1)
            # cv2.imshow('iris', zeros)
            #
            # self.test_in_big(thresh, cnt_area[0])

            # 0 : size, 1 : x, 2 : y
            return cnt_area[0][1:]

        else:
            return cnt_area

    def makeTrackingObj(self, img, faceBox) :
        x, y, w, h = faceBox
        # cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

        faceROI = img[y:y + h, x:x + w]
        # cv2.imshow('first_face', faceROI)

        # 트랙커 객체 생성 ---⑤
        self.tracker = self.trackerAlg()
        isInit = self.tracker.init(img, faceBox)

        # isInit : tracking Success / Fail
        if isInit is True :
            eyeBox = self.SearchEye(faceROI)

            if len(eyeBox) != 4:
                self.Reset()
                return

            ex, ey, ew, eh = eyeBox
            # cv2.rectangle(faceROI, (ex, ey), (ex + ew, ey + eh), (255, 0, 0), 2)

            # eyeROI = faceROI[ey: ey + eh, ex: ex + ew]
            # cv2.imshow('firse_eye', eyeROI)

            # 트랙커 객체 생성 ---⑤
            self.eyetracker = self.trackerAlg()
            eye_isInit = self.eyetracker.init(faceROI, eyeBox)

            # isInit : tracking Success / Fail
            if eye_isInit is True :
                self.isFirst = False
            else :
                # print('Cannot find Eyes')
                self.Reset()

    def trackingFace(self, img) :
        # Face Tracking
        height, width, _ = img.shape
        ok, faceBox = self.tracker.update(img)  # 새로운 프레임에서 추적 위치 찾기 ---③
        if ok:  # 추적 성공
            faceBox = list(map(int, faceBox))
            x, y, w, h = faceBox

            # cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            # x < 0, y < 0, x + w > width, y + h > height -> reset
            # 얼굴이 화면을 벗어나면 초기화
            if x <= 0 or y <= 0 or x + w >= width or y + h >= height:
                # print("ROI is escape from cam!!!")
                return []
            else:
                faceROI = img[y:y + h, x:x + w]
                return faceROI
        else :
            # print('Face Tracking Fail !!!')
            return []

    def trackingEyes(self, faceROI) :
        ok2, eyesBox = self.eyetracker.update(faceROI)

        if ok2:  # 추적 성공
            self.failCnt = 0

            eyesBox = list(map(int, eyesBox))
            x, y, w, h = eyesBox
            # cv2.rectangle(faceROI, (x, y), (x + w, y + h), (255, 0, 0), 2)
            eyeROI = faceROI[y:y + h, int(x + w * 0.1): int(x + w * 0.9)]

            return eyeROI

        else :
            if self.failCnt > 10 :
                # print('Fail Eyes ROI Tracking!!!')
                self.Reset()
            self.failCnt += 1
            return []

    def get_pos_L(self, thresh) :
        box = self.get_eye_shape(thresh)

        if not box or box[0] == 0:
            return

        # first execute check
        if self.LIrisFirst is True :
            self.LIrisFirst = False
            self.LIrisbox = box

        # fix box size
        if self.LIrisbox[2] > box[2] :
            x, y, w, h = self.LIrisbox
        else :
            x, y, w, h = box

        if box[3] > self.LIrisbox[3] :
            self.upsig = True

        thresh = thresh[y:y + h, x:x + w]

        # cv2.imshow('thresh-left', thresh)

        kernel = np.ones((9, 9), np.uint8)
        result = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

        pos = self.getCenter(result)

        return pos

    def get_pos_R(self, thresh) :
        box = self.get_eye_shape(thresh)

        if not box or box[0] == 0:
            return

        if self.RIrisFirst is True :
            self.RIrisFirst = False
            self.RIrisbox = box

        if self.RIrisbox[2] > box[2] :
            x, y, w, h = self.RIrisbox
        else :
            x, y, w, h = box

        if box[3] > self.RIrisbox[3] :
            if self.upsig :
                self.upsig = True
            else :
                self.upsig = False

        thresh = thresh[y:y + h, x:x + w]

        # cv2.imshow('thresh-right', thresh)

        kernel = np.ones((9, 9), np.uint8)
        result = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

        # cv2.imshow('thresh-right after open', result)
        pos = self.getCenter(result)

        return pos

    def setting(self, now) :
        # 이전 위치와 현재위치의 차이가 적으면 움직임이 없다
        # 오른쪽 아래를 일정시간 이상 응시하면 세팅완료

        # 일정시간 초과하여 세팅값 저장
        if self.setCnt > 10 :
            # list average, axis=0 -> column average
            self.irisRD = np.mean(self.RDList, axis=0, dtype=np.int64)

            self.isSet = True

            if self.irisRD[0] > self.irisCC[0] and self.irisRD[1] > self.irisCC[1] :
                self.detectROI = np.zeros(shape=((self.irisRD[0] - self.irisCC[0]) * 2, (self.irisRD[1] - self.irisCC[1]) * 2),
                                 dtype=np.int8)
            else :
                self.Reset()

        elif self.setCnt == 0 :
            self.irisCC = now

        # 이전과 현재의 차이가 적으면 움직임이 없는것으로 판단
        if -4 < self.irisRD[0] - now[0] < 4 and -2 < self.irisRD[1] - now[1] < 2 :
            self.setCnt += 1
            self.RDList = np.append(self.RDList, np.array([now]), axis=0)
        # 차이가 많으면 초기화
        else:
            self.irisRD = now
            self.setCnt = 1

    def DetectMove(self, pos) :
        width, height = self.detectROI.shape
        move = ""

        print(self.detectROI.shape)
        print(pos)

        optimR = width/2 + width*0.05
        optimL = width/2 - width*0.05
        optimD = height/2 + height*0.025
        optimU = height/2 - height*0.025

        if pos[0] > optimR :
            self.detectLR = "r"
        elif pos[0] < optimL :
            self.detectLR = "l"
        else :
            pass

        if pos[1] > optimD :
            self.detectUD = "d"
        elif pos[1] < optimU :
            self.detectUD = "u"
        elif self.upsig :
            self.detectUD = "u"
        else :
            pass

        move = self.detectLR + self.detectUD

        return move

    def trackingIris(self, eyeROI) :
        eyeROI = cv2.resize(eyeROI, dsize=(0, 0), fx=3, fy=3, interpolation=cv2.INTER_LINEAR)
        height, width, _ = eyeROI.shape
        pos = []

        eyeROIGray = cv2.cvtColor(eyeROI, cv2.COLOR_BGR2GRAY)
        ROI_equ = cv2.equalizeHist(eyeROIGray)
        # threshold value -> 돌릴수있는 핀(?) or 트랙바로 처리
        _, thresh = cv2.threshold(ROI_equ, threshold_val, 255, cv2.THRESH_BINARY_INV)

        thresh_left = thresh[0: height, 0: width // 2]
        thresh_right = thresh[0: height, width // 2: width]

        left_pos = self.get_pos_L(thresh_left)
        right_pos = self.get_pos_R(thresh_right)

        if right_pos and left_pos :
            left_pos = list(map(int, left_pos))
            right_pos = list(map(int, right_pos))

            pos = [int((left_pos[0] + right_pos[0])/2), int((left_pos[1] + right_pos[1])/2)]

        # only right sight test
        # thresh_right = thresh[0: height, width // 2: width]
        #
        # right_pos = self.get_pos_R(thresh_right)
        #
        # if right_pos :
        #     right_pos = list(map(int, right_pos))
        #
        #     pos = right_pos

        return pos


    def Reset(self) :
        EyeT.__init__(self)
        cv2.destroyAllWindows()

    def main(self, img) :
        sendmsg = ""
        if self.isFirst is True :
            faceBox = self.SearchFace(img)

            if len(faceBox) != 4:
                # print('cannot search Face !!!')
                return

            if self.faceCount > 30 :
                self.makeTrackingObj(img, faceBox)
                sendmsg = "click"

            self.faceCount += 1

        else :
            faceROI = self.trackingFace(img)
            if len(faceROI) != 0 :
                eyeROI = self.trackingEyes(faceROI)

                if len(eyeROI) != 0 :
                    center_pos = self.trackingIris(eyeROI)

                    if len(center_pos) != 0 :
                        if self.isSet is True:
                            # 좌표에 따른 움직임 감지
                            sendmsg = self.DetectMove(center_pos)
                            cv2.putText(img, str(sendmsg), (10, 110), cv2.FONT_HERSHEY_COMPLEX, img.shape[1] / 500,
                                        (0, 0, 255), 2)
                        else :
                            self.setting(center_pos)
                            cv2.putText(img, "see right_under", (10, 110), cv2.FONT_HERSHEY_COMPLEX, img.shape[1] / 500,
                                        (0, 0, 255), 2)
                            if self.isSet is True :
                                sendmsg = "click"

            else :
                self.Reset()
                sendmsg = "fail"

        return sendmsg