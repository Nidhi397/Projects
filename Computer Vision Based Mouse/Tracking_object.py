import cv2
import numpy as np

cam=cv2.VideoCapture(0)
lower_white=np.array([0,0,220])
upper_white=np.array([172,111,255])
while(1):
    ret,frame=cam.read()
    
    #Smoothen image
    img_smooth=cv2.GaussianBlur(frame,(7,7),0)

    #Threshold image for white color

    image_hsv=cv2.cvtColor(img_smooth,cv2.COLOR_BGR2HSV)
    img_threshold=cv2.inRange(image_hsv,lower_white,upper_white)
    contours,hierarchy=cv2.findContours(img_threshold,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)

    #Find index of largest contour
    
    if(len(contours)!=0):
       areas=[cv2.contourArea(c) for c in contours]
       
        
       max_index=np.argmax(areas)
       cnt=contours[max_index]
       x,y,w,h=cv2.boundingRect(cnt)
       cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),3)
    cv2.imshow("Frame",frame)

    key=cv2.waitKey(10)
    if cv2.waitKey(1) & 0xFF==ord('q'):
        break


cam.release()
cv2.destroyAllWindows()
    
