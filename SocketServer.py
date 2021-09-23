from socket import *
import sys
import cv2
import time
from socket import *

HOST = '127.0.0.1'
PORT = 10000
ADDR = (HOST, PORT)

# 소켓 생성
serverSocket = socket(AF_INET, SOCK_STREAM)
# 소켓 주소 정보 할당
serverSocket.bind(ADDR)
# 연결 수신 대기 상태
serverSocket.listen(100)
print("Listen.....")
# 연결 수락
clientSocket, addr_info = serverSocket.accept()

try:
    while True:
        msg = clientSocket.recv(10).decode()

        print("손가락 개수 : " + msg[0])

        if msg[1] == "0":
            print("클릭 X ")
        else:
            print("클릭 O")

        if msg[2] == "-1":
            print("방향 : 왼쪽")
        elif msg[2] == "1":
            print("방향 : 오른쪽")
        else:
            print("방향 : 중간")

except Exception as e:
    print('%s:%s' % ADDR)
    # 소켓 종료
    clientSocket.close()
    serverSocket.close()
    print('close')
    sys.exit()

# 소켓 종료
clientSocket.close()
serverSocket.close()
print('close')
