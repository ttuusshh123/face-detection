import cv2
import numpy as np
import matplotlib.pyplot as plt


face = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')




def detect_face(frame):
    face_img = frame.copy()
    face_rects = face.detectMultiScale(face_img)
    for (x,y,w,h) in face_rects:
        cv2.rectangle(face_img, (x,y), (x+w, y+h), (255,0,0),2 )
    return face_img


cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    cv2.imshow('x',detect_face(frame))
    k = cv2.waitKey(1)
    if k==ord('p'):
        break

cv2.destroyAllWindows()
