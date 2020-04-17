import cv2
import numpy as np
import matplotlib.pyplot as plt


face = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

img = cv2.imread('images.jfif')






def detect_face(frame):
    face_img = frame.copy()
    face_rects = face.detectMultiScale(face_img)
    for (x,y,w,h) in face_rects:
        cv2.rectangle(face_img, (x,y), (x+w, y+h), (255,0,0),2 )
    return face_img



while True:

    cv2.imshow('x',detect_face(img))
    k = cv2.waitKey(1)
    if k==0xFF:
        break

cv2.destroyAllWindows()

