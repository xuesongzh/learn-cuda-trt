import cv2
import os

RTSP_URL = "rtsp://admin:SEGJKL@192.168.1.185:554/h264/ch1/main/av_stream"


cap = cv2.VideoCapture(RTSP_URL)

if not cap.isOpened():
    print('Cannot open RTSP stream')
    exit(-1)

while True:
    success, img = cap.read()
    img = cv2.resize(img, (640, 480))
    cv2.imshow('RTSP raw', img)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):  # Keep running until you press `q`
        cap.release()
        break

cv2.destroyAllWindows()