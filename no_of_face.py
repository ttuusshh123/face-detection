import cv2
import numpy as np
import matplotlib.pyplot as plt


face = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

img = cv2.imread('images.jfif')

def detect_faces(img):
    img_copy = img.copy()
    face_rect = face.detectMultiScale(img_copy)
    for (x,y,w,h) in face_rect:
        cv2.rectangle(img_copy, (x,y), (x+w, y+h), (255,0,0),2 )
    return img_copy,len(face_rect)

# Window name in which image is displayed 
window_name = 'Image'
  
# font 
font = cv2.FONT_HERSHEY_SIMPLEX 
  
# org 
org = (2, 20) 
  
# fontScale 
fontScale = 0.5
   
# Blue color in BGR 
color = (255,255,0) 
  
# Line thickness of 2 px 
thickness = 2

while True:
    temp,x = detect_faces(img)
    st = f"there are {x} faces"
    image = cv2.putText(temp, st, org, font,  
                   fontScale, color, thickness, cv2.LINE_AA) 
    cv2.imshow('x', temp)
    
    k = cv2.waitKey(1)
    if k==ord('q'):
        break
cv2.destroyAllWindows()