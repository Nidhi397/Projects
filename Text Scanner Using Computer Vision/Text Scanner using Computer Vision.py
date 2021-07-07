import cv2 
import numpy as np
import pytesseract as pyt

from tkinter.filedialog import askopenfilename
filename=askopenfilename()

pyt.pytesseract.tesseract_cmd='/usr/local/Cellar/tesseract/4.1.1/bin/tesseract'
img=cv2.imread(filename)

image_w=np.copy(img)

#Find Contour of the Image
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

#Smoothening the grayscale image using Gaussian Blur
smooth_img=cv2.GaussianBlur(gray,(5,5),0)

thresh=cv2.Canny(smooth_img,150,300)

contours,hierarchy=cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)

if (len(contours)!=0):
    areas=[cv2.contourArea(c) for c in contours]
    max_index=np.argmax(areas)
    cnt=contours[max_index]
epsilon=0.05*cv2.arcLength(cnt,True)
approx=cv2.approxPolyDP(cnt,epsilon,True)

cv2.drawContours(image_w,[approx],-1,(255,0,0),3)
cv2.imshow("image",image_w)
cv2.waitKey(0)

#Perspective Transformation Matrix and unwarp
src=np.float32([approx[0,0],approx[1,0],approx[2,0],approx[3,0]])
dst=np.float32([[0,0],[0,300],[500,300],[500,0]])

M=cv2.getPerspectiveTransform(src,dst)

warped=cv2.warpPerspective(img,M,(500,300))
cv2.imshow("warped",warped)
cv2.waitKey(0)

#Pre-Process Image for Extracting Test
resize=cv2.resize(warped,None,fx=8,fy=8,interpolation=cv2.INTER_CUBIC) #Resize to zoom in to text
warp_gray=cv2.cvtColor(resize,cv2.COLOR_BGR2GRAY)  #Converting the BGR image to grayscale 

ret,thresh_warp=cv2.threshold(warp_gray,150,255,1) #Inverted Binary thresholding to get white text on black background 

        



cv2.destroyAllWindows()

text=pyt.image_to_string(thresh_warp)
print(text)
