import cv2 
import matplotlib.pyplot as plt
import numpy
%matplotlib inline


def display(img,cmap='gray'):
    plt.figure(figsize=(10,8))
    plt.imshow(img,cmap='gray')

nadia = cv2.imread('naida_murad.jpeg',0)

display(nadia)

face_cascade  =cv2.CascadeClassifier('DATA/haarcascades/haarcascade_frontalface_default.xml')

def detectFace(img):
    
    face_img = img.copy()
    face_retangle = face_cascade.detectMultiScale(face_img,scaleFactor=1.2,minNeighbors=5)
    
    for (x,y,w,h) in face_retangle:
        cv2.rectangle(face_img,(x,y),(x+w,y+h),(255,255,255),1)
        
    return face_img

nadia_rec = detectFace(nadia)
display(nadia_rec)


# Detecting Face from Video 

cap = cv2.VideoCapture(0)

while True:
    
    ret ,frame = cap.read(0)
    
    frame = detectFace(frame)
    
    cv2.imshow('DETECT FACE WITH VIDEO',frame)
    
    k = cv2.waitKey(1)
    
    if k == 27:
        break
        
cap.release()
cv2.destroyAllWindows()