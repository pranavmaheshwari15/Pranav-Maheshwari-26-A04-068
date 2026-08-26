import cv2
import numpy as np
def canny(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)     #Converting image color to gray
    blur = cv2.GaussianBlur(gray, (5,5), 0)     #To reduce noise in image
    canny = cv2.Canny(blur,50,150)      #Traces edges which correspond to maximum change in brightness
    return canny

image = cv2.imread("Input/1.png")  #To load the image
lane_image = np.copy(image)     #Copying the array of image
canny = canny(lane_image)

cv2.imshow('result',canny)  #To read image
cv2.waitKey(0)   #Display result window until we press anything on keyboard
