# 读取摄像头并显示
import cv2
import numpy as np

cap = cv2.VideoCapture(2)
while (True):
    # 读取一帧
    ret, frame = cap.read()
    # flip
    frame = cv2.flip(frame, 1)
    # 显示图像
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# 释放资源
cap.release()