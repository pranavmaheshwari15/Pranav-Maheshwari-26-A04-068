import cv2
import numpy as np
import matplotlib.pyplot as plt
def canny(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)     #Converting image color to gray
    blur = cv2.GaussianBlur(gray, (5,5), 0)     #To reduce noise in image
    canny = cv2.Canny(blur,50,150)      #Traces edges which correspond to maximum change in brightness
    return canny

def region_of_interest (image):
    height = image.shape[0]
    polygons = np.array([[(-175,height),(1150,height),(575,225)]])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask,polygons,255)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image

def display_lines(image, lines):
    line_image = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            x1,y1,x2,y2 = line.reshape(4)      #Reshape to 1D aray with 4 elemengs
            cv2.line(line_image, (x1,y1),(x2,y2), (255,0,0), 10)
    return line_image

image = cv2.imread("Input/1.png")  #To load the image
lane_image = np.copy(image)     #Copying the array of image
canny_image = canny(lane_image)
cropped_image = region_of_interest(canny_image)
lines = cv2.HoughLinesP(cropped_image, 2, np.pi/180, 100, np.array([]), minLineLength = 40, maxLineGap = 5)
line_image = display_lines(lane_image, lines)
combo_image = cv2.addWeighted(lane_image,0.8, line_image,2,1)       #MUltiplies array with 0.8 decreasing intensity

cv2.imshow("result",combo_image)    #To read image
cv2.waitKey(0)   #Display result window until we press anything on keyboard
