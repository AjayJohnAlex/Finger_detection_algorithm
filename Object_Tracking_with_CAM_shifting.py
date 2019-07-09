import numpy as np
import cv2

face_cascade = cv2.CascadeClassifier('DATA/haarcascades/haarcascade_frontalface_default.xml')


cap = cv2.VideoCapture(0)

ret, frame  = cap.read()

face_cascade = cv2.CascadeClassifier('DATA/haarcascades/haarcascade_frontalface_default.xml')
face_rectangle = face_cascade.detectMultiScale(frame,scaleFactor=1.3,minNeighbors=5)

# face_rectangle[0] is taken coz we need only the starting frame for rest of the tracking
(face_x,face_y,w,h) = tuple(face_rectangle[0])
track_window = (face_x,face_y,w,h)

# getting the region of interest
roi  = frame[face_y:face_y+h,face_x:face_x+w]
# converting to hsv color graph
hsv_roi = cv2.cvtColor(roi,cv2.COLOR_BGR2HSV)
#histograph for saturation and coloration of the HSV
roi_hist = cv2.calcHist([hsv_roi],[0],None,[180],[0,180])
#normalise the histogram
roi_norm = cv2.normalize(roi_hist,roi_hist,0,255,cv2.NORM_MINMAX)
#going for 10 iterations or atleast one
termination_criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,10,1)

while True:
    
    ret, frame = cap.read()
    
    if ret == True:
        
        hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
        
        dst = cv2.calcBackProject([hsv],[0],roi_hist,[0,180],1)
        
        ret, track_window = cv2.CamShift(dst,track_window,termination_criteria)
        pts = cv2.boxPoints(ret)
        pts = np.int0(pts)
        img2 = cv2.polylines(frame,[pts],True,(0,0,255),2)
        
        cv2.imshow('CAMshift',img2)
    
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break
    else:
        break
        
        
cap.release()
cv2.destroyAllWindows()