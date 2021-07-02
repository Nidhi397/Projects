import cv2
import numpy as np

cam=cv2.VideoCapture(0)
lower_white=np.array([0,0,220])
upper_white=np.array([172,111,255])
while(1):
    ret,frame=cam.read()
    frame=cv2.flip(frame,1)
    w=frame.shape[1]
    h=frame.shape[0]
    #Smoothen image
    img_smooth=cv2.GaussianBlur(frame,(7,7),0)

    #Define ROI
    mask=np.zeros_like(frame)
    mask[50:350,50:350]=[255,255,255]
    img_roi=cv2.bitwise_and(img_smooth,mask)
    cv2.rectangle(frame,(50,50),(350,350),(0,255,0),2)
    cv2.line(frame,(150,50),(150,350),(0,255,0),1)
    cv2.line(frame,(250,50),(250,350),(0,255,0),1)
    cv2.line(frame,(50,150),(350,150),(0,255,0),1)
    cv2.line(frame,(50,250),(350,250),(0,255,0),1)



    


    #Threshold image for white color

    image_hsv=cv2.cvtColor(img_roi,cv2.COLOR_BGR2HSV)
    img_threshold=cv2.inRange(image_hsv,lower_white,upper_white)
    contours,hierarchy=cv2.findContours(img_threshold,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)

    #Find index of largest contour
    
    if(len(contours)!=0):
       areas=[cv2.contourArea(c) for c in contours]
       
        
       max_index=np.argmax(areas)
       cnt=contours[max_index]

       #Pointer on video
       M=cv2.moments(cnt)
       if (M['m00']!=0):
           cx=int(M['m10']/M['m00'])
           cy=int(M['m01']/M['m00'])
           cv2.circle(frame,(cx,cy),4,(0,255,0),-1)

           #Cursor Motion
           if cx in range(150,250):
               if cy<150:
                   print("Upper Middle")
               elif cy>250:
                   print("Lower Middle")

               else:
                   print("Centre")
           if cy in range(150,250):
               if cx<150:
                   print("Left Middle")
               elif cx>250:
                   print("Right Middle")
               else:
                   print("Centre")

    
    cv2.imshow("Frame",frame)

    key=cv2.waitKey(10)
    if cv2.waitKey(1) & 0xFF==ord('q'):
        break


cam.release()
cv2.destroyAllWindows()
    
