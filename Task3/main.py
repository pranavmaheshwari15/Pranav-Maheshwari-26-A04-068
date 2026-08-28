import cv2
import numpy as np

def region_of_interest(image):
    height, width = image.shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)
    top_crop = int(height * 0.02)           #Ignoring the top 2% of image used to ignore sky,overlays
    bottom_crop = int(height * 0.95)        #USed to ignore bottom 5% of the image used to ignore dashboards
    cv2.rectangle(mask, (0, top_crop), (width, bottom_crop), 255, -1)       #-1 is used to fill the rectangle
    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    return cv2.bitwise_and(image, mask)

def find_potholes(image, min_area=120, max_area=None):
    h, w = image.shape[:2]
    if max_area is None:
        max_area = 0.05 * h * w

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, white_mask = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)        #"_" is used to make unwanted variable throwaway variable; 240 is set as threshold value; thresh_binary converts grayscale image into black and white using threshold value i.e convert into white if pixel > 240 and black if pixel<240
    contours, _ = cv2.findContours(
        white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE      #"_" is used to remove contour hierarchy; white_mask is used to input 8 bit single channel image; retr is used to only take care of outer perimeter;chain is used to take care only of 4 corner points
    )
    #Returns a 2D numpy array
    detections = []
    for cnt in contours:
        area = cv2.contourArea(cnt)     #contour area represents a shoelace formula to calculate area using 2D array
        if area < min_area or area > max_area:
            continue
        x, y, w_box, h_box = cv2.boundingRect(cnt)
        hull = cv2.convexHull(cnt)      #used to higlight the tighest boundary around contour
        hull_area = cv2.contourArea(hull)       #to calculate area of boundaru
        solidity = float(area) / hull_area if hull_area > 0 else 0      #If rubber band is stretched around contour the extent of strectch is called solidity for instance circles have high solidity of around 1 as compared to irreruglar rectangles in which rubber band is stretched to intricate boundaries
        bbox_area = w_box * h_box
        extent = float(area) / bbox_area if bbox_area > 0 else 0
        if solidity < 0.80 or extent < 0.55 or h_box < 10:      #standard values
            continue
        aspect_ratio = max(w_box, h_box) / float(max(1, min(w_box, h_box)))     #used to calculate the proportions to identify objects
        if aspect_ratio < 4.0:      #standard value
            detections.append((x, y, w_box, h_box, "Pothole"))
    return detections

def split_merged_contour(image, x, y, w_box, h_box):        #used to separate two touching potholes or obstacles
    roi_crop = image[y : y + h_box, x : x + w_box]      #cropes to only bounding box area
    vert_hist = np.sum(roi_crop > 0, axis=0)        #sums non zero pixels vertically down each column to convert 2d to 1d array
    smoothed_hist = cv2.GaussianBlur(
        vert_hist.astype(np.float32).reshape(1, -1), (1, 15), 0     #np.float and reshape convert 1d array to 2d format; rest to blur the image to smoothen soft and unwanted edges; flatten to again convert to 1d
    ).flatten()
    margin = int(w_box * 0.20)  
    search_region = smoothed_hist[margin : w_box - margin]
    if len(search_region) == 0:
        return [(x, y, w_box, h_box, "Obstacle")]
    min_idx = margin + np.argmin(search_region)
    min_val = smoothed_hist[min_idx]
    max_val = np.max(smoothed_hist)
    if min_val < 0.85 * max_val and w_box > 1.1 * h_box:
        box1 = (x, y, min_idx, h_box, "Obstacle")
        box2 = (x + min_idx, y, w_box - min_idx, h_box, "Obstacle")
        return [box1, box2]
    return [(x, y, w_box, h_box, "Obstacle")]


def find_all_standing_obstacles(image, min_area=150, max_area=None):
    h, w = image.shape[:2]
    if max_area is None:
        max_area = 0.40 * h * w
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)        #Converts RGB to HSV as it is better
    lower_blue = np.array([100, 40, 10])        #Blue standard threshold
    upper_blue = np.array([135, 255, 255])
    blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)        #Converts blue to white and verything else black
    lower_yellow = np.array([15, 60, 60])       #Standard yellow thresholds
    upper_yellow = np.array([38, 255, 255])
    yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)      #Converts yellow to white and everything else black
    lower_green = np.array([35, 40, 40])
    upper_green = np.array([85, 255, 255])
    green_mask = cv2.inRange(hsv, lower_green, upper_green)
    combined_mask = cv2.bitwise_or(blue_mask, cv2.bitwise_or(yellow_mask,green_mask))      #Combines in single image
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 9))
    clean_mask = cv2.morphologyEx(
        combined_mask, cv2.MORPH_CLOSE, kernel, iterations=2
    )
    contours, _ = cv2.findContours(
        clean_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    detections = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area or area > max_area:
            continue
        x, y, w_box, h_box = cv2.boundingRect(cnt)
        cnt_mask = np.zeros_like(image)     #Creates black image of same detections
        cv2.drawContours(cnt_mask, [cnt], -1, 255, -1)    #Creates given contours in solid white on black image
        split_boxes = split_merged_contour(cnt_mask, x, y, w_box, h_box)
        detections.extend(split_boxes)
    return detections

def remove_overlaps(base_detections, new_detections):
    def iou(a, b):      #Intersection over unioin as to how much two boxes overlap between 0 to 1
        ax1, ay1, aw, ah = a[:4]        #Top left coordinates and width and height
        bx1, by1, bw, bh = b[:4]
        ax2, ay2, bx2, by2 = ax1 + aw, ay1 + ah, bx1 + bw, by1 + bh     #Bottom right
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)     #Top left overlapping
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)     #Bottom right overlapping
        iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
        inter = iw * ih
        union = aw * ah + bw * bh - inter
        return inter / union if union > 0 else 0
    result = []
    for nd in new_detections:
        if all(iou(nd, bd) < 0.20 for bd in base_detections):
            result.append(nd)
    return result


CLASS_COLORS = {
    "Pothole": (0, 0, 255),
    "Obstacle": (0, 140, 255),
}


def draw_obstacles_on_image(image, detections):
    annotated = image.copy()
    for i, (x, y, w, h, cls) in enumerate(detections, start=1):
        color = CLASS_COLORS.get(cls, (0, 0, 255))
        cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)
        label = f"#{i} {cls} ({x},{y})"
        text_y = y - 8 if y - 8 > 10 else y + h + 18
        cv2.putText(
            annotated,
            label,
            (x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
            cv2.LINE_AA,
        )
    return annotated

image_path = [
    "Input/1.png",
    "Input/2.png",
    "Input/3.png",
    "Input/4.png",
    "Input/5.png",
    "Input/6.png",
    "Input/7.png",
    "Input/8.png",
]
for j in range (8):
    image = cv2.imread(image_path[j])
    pothole_image = np.copy(image)

    roi_bgr = region_of_interest(pothole_image)
    potholes = find_potholes(roi_bgr)

    standing_obstacles = find_all_standing_obstacles(roi_bgr)
    standing_obstacles = remove_overlaps(potholes, standing_obstacles)

    boxes = potholes + standing_obstacles
    combo_image = draw_obstacles_on_image(pothole_image, boxes)

    count_text = f"Total obstacles/potholes detected: {len(boxes)}"
    cv2.putText(
        combo_image,
        count_text,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2,
    )

    print(count_text)
    for i, (x, y, w, h, cls) in enumerate(boxes, start=1):
        print(f"  #{i} [{cls}]: top-left=({x},{y})  width={w}  height={h}")

    cv2.imshow(f"Output - {j+1}", combo_image)
    cv2.imwrite(f"Output/Output - {j+1}.png", combo_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()