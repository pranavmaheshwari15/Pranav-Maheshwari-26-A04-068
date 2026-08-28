import cv2
import numpy as np
import matplotlib.pyplot as plt
def canny(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)     #Converting image color to gray
    blur = cv2.GaussianBlur(gray, (5,5), 0)     #To reduce noise in image
    canny = cv2.Canny(blur,50,150)      #Traces edges which correspond to maximum change in brightness
    return canny

polygons = [[(-175, 0), (1150, 0), (575, 225)],
            [(-175, 0), (1150, 0), (575, 225)],
            [(-175, 0), (1150, 0), (575, 225)],
            [(-200, 0), (400, 0), (225,130)],
            [(-400, 0), (500, 0), (200, 200)]]

y2_ = [390,390,390,175,300]

slope_threshold_min = [-0.5,-0.5,-0.5,0,-0.5]
slope_threshold_max = [0.5,0.5,0.5,0,0.5]

image_path = [
    "C:/Users/prana/Desktop/Programming/UGV/Pranav-Maheshwari-26-A04-068/Task2/Input/1.png",
    "C:/Users/prana/Desktop/Programming/UGV/Pranav-Maheshwari-26-A04-068/Task2/Input/2.png",
    "C:/Users/prana/Desktop/Programming/UGV/Pranav-Maheshwari-26-A04-068/Task2/Input/3.png",
    "C:/Users/prana/Desktop/Programming/UGV/Pranav-Maheshwari-26-A04-068/Task2/Input/4.jpeg",
    "C:/Users/prana/Desktop/Programming/UGV/Pranav-Maheshwari-26-A04-068/Task2/Input/5.jpeg"
]

for i in range (5):

    def region_of_interest (image):
        height = image.shape[0]
        mask = np.zeros_like(image)
        x,y = polygons[i][0]
        y = height
        polygons[i][0] = x,y
        a,b = polygons[i][1]
        b = height
        polygons[i][1] = a,b
        polygons[i] = np.array([polygons[i]])
        cv2.fillPoly(mask,polygons[i],255)
        masked_image = cv2.bitwise_and(image, mask)
        return masked_image

    def display_lines(image, lines):
        line_image = np.zeros_like(image)
        if lines is not None:
            for line in lines:
                x1,y1,x2,y2 = line.reshape(4)      #Reshape to 1D aray with 4 elemengs
                cv2.line(line_image, (x1,y1),(x2,y2), (0,255,0), 10)
        return line_image

    def make_coordinates(image,line_parameters):
        slope,intercept = line_parameters
        y1 = image.shape[0]
        y2 = y2_[i]
        x1 = int((y1 - intercept)/slope)
        x2 = int((y2-intercept)/slope)
        return np.array ([x1,y1,x2,y2])

    def average_slope_intercept(image, lines):
        left_fit = []
        right_fit = []
        if lines is None:
            return np.array([])
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            parameters = np.polyfit((x1, x2), (y1, y2), 1)
            slope = parameters[0]
            intercept = parameters[1]
            if slope < slope_threshold_min[i] :
                left_fit.append((slope, intercept))
            elif slope > slope_threshold_max[i]:
                right_fit.append((slope, intercept))
        averaged_lines = []
        if len(left_fit) > 0:
            left_fit_average = np.average(left_fit, axis=0)
            left_line = make_coordinates(image, left_fit_average)
            averaged_lines.append(left_line)
        if len(right_fit) > 0:
            right_fit_average = np.average(right_fit, axis=0)
            right_line = make_coordinates(image, right_fit_average)
            averaged_lines.append(right_line)
        return np.array(averaged_lines)

    image = cv2.imread(image_path[i])  #To load the image
    lane_image = np.copy(image)     #Copying the array of image
    canny_image = canny(lane_image)
    cropped_image = region_of_interest(canny_image)
    lines = cv2.HoughLinesP(cropped_image, 2, np.pi/180, 100, np.array([]), minLineLength = 40, maxLineGap = 20)
    averaged_lines = average_slope_intercept(lane_image,lines)
    line_image = display_lines(lane_image, averaged_lines)
    combo_image = cv2.addWeighted(lane_image,0.8, line_image,3,1)       #MUltiplies array with 0.8 decreasing intensity

    cv2.imshow(f"Output - {i+1}", combo_image)    #To read image
    cv2.imwrite(f"Output/output_{i+1}.png", combo_image)
    cv2.waitKey(0)   #Display result window until we press anything on keyboard
