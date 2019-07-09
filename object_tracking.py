# OBJECT TRACKING WITH API

import cv2

print('ENTER YOUR TRACKING CHOICE')
print('''
         Enter 0 : BOOSTING,
         Enter 1 : MIL
         Enter 2 : KCL
         Enter 3 : TLD
         Enter 4 : MEDIAN FLOW''')

actual_choice = input('Enter your choice: ')
# if actual_choice not in (0,1,2,3,4):
#     print('Sorry Invalid choice')

def tracker_choice(choice):
    
    if choice == '0':
        tracker = cv2.TrackerBoosting_create()
    elif choice == '1':
        tracker = cv2.TrackerMIL_create()
    elif choice == '2':
        tracker = cv2.TrackerKCF_create()
    elif choice == '3':
        tracker = cv2.TrackerTLD_create()
    elif choice == '4':
        tracker = cv2.TrackerMedianFlow_create()
        
    return tracker

type_tracker = tracker_choice(actual_choice)
tracker_name = str(type_tracker).split()[0][1:]

cap = cv2.VideoCapture(0)

ret ,frame = cap.read()
# SELECT THE ROI MANUALLY
roi = cv2.selectROI(frame,False)
#INTIALISE THE TRACKER WITH FIRST FRAME and BOUNDING BOX
ret = type_tracker.init(frame,roi)

while True:
    # NEW FRAME
    ret ,frame = cap.read()
    
    success,roi = type_tracker.update(frame)
    
    #tuple unpacking
    (x,y,w,h) = tuple(map(int,roi))
    
    if success:        
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)
        
    else:
        cv2.putText(frame,'SORRY CANT FIND THE OBJECT',(10,500),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)
    
    cv2.imshow(tracker_name,frame)
    
    k = cv2.waitKey(1) & 0xFF
    if k ==27:
        break    
    
    
cap.release()
cv2.destroyAllWindows()