from socket import *

# 소켓 통신
HOST = '127.0.0.1'
PORT = 10000

ADDR = (HOST, PORT)

clientSocket = socket(AF_INET, SOCK_STREAM)  # 서버에 접속하기 위한 소켓을 생성한다.
clientSocket.connect(ADDR)  # 서버에 접속을 시도한다.

def send(msg) :
    if (isSend is True) and FPScnt >= 60:
        clientSocket.send(msg.encode())
        isSend = False
        FPScnt = 0

    if msg == "000":
        print("move")
        isSend = True

def sendS() :
    print('q')