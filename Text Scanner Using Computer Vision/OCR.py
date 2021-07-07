import cv2
import numpy as np
import pytesseract as pyt

pyt.pytesseract.tesseract_cmd = '/usr/local/Cellar/tesseract/4.1.1/bin/tesseract'
img=cv2.imread('image.png')
text=pyt.image_to_string(img)
print(text)

cv2.imshow('Original',img)

cv2.waitKey(0)
cv2.destroyAllWindows()
