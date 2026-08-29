import cv2
import numpy as np
import heapq
import math

image_path = [
    "Input/1.jpeg",
    "Input/2.jpeg"
]
checkpoints = [

    [# START
    (1000, 733),

    # Right side → upper-right
    (973, 551),
    (930, 450),
    (830, 250),

    # Upper section
    (720, 250),
    (680, 410),
    (600, 430),

    # Upper-left
    (430, 410),
    (330, 300),

    # Left side
    (250, 370),
    (250, 580),
    (300, 730),

    # Lower-left
    (300, 900),
    (440, 910),
    (520, 850),

    # Lower section
    (660, 900),
    (790, 850),
    (880, 830),
    (900, 730),

    # Return to START
    (1000, 700)],
    [(1030,715),

     (1060,560),
     (1015,420),
     (910,290),

     (845,300),
     (690,210),
     (585,180),

     (535,270),
     (320,290),
     (220,280),

     (290,430),
     (500,580),
     (260,720),

     (190,800),
     (265,920),
     (430,965),
     (560,1020),


     (725,925),
     (880,980),
     (940,800),

     (1030,715)]
]

for j in range (len(image_path)):
    image = cv2.imread(image_path[j])
    output = image.copy()
    height, width = image.shape[:2]

    road_mask = np.zeros((height, width),dtype=np.uint8)
    points = np.array(checkpoints[j],dtype=np.int32)       #To convert checkpoints to numpy array
    cv2.polylines(road_mask,[points],False,255,thickness=150)       #Line joing the points
    kernel = np.ones((15, 15),np.uint8)     # Fill small gaps in the road mask.
    road_mask = cv2.morphologyEx(road_mask,cv2.MORPH_CLOSE,kernel)

    hsv = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)     #Converting to hsv as converting to gray wont be helpful
    obstacle_mask = cv2.inRange(hsv,np.array([0, 70, 30]),np.array([179, 255, 255]))        # Detect pixels with reasonably high saturation.Gray areas generally have low saturation. Colored obstacles have higher saturation.
    kernel = np.ones((7, 7),np.uint8)       # Remove small noise.
    obstacle_mask = cv2.morphologyEx(obstacle_mask,cv2.MORPH_CLOSE,kernel)

    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    circles = cv2.HoughCircles(gray,cv2.HOUGH_GRADIENT,dp=1.2,minDist=50,param1=80,param2=25,minRadius=12,maxRadius=30)     # Detect circles; param 1 canny edge threshold;param 2 canny detection threshold
    if circles is not None:
        circles = np.round(circles[0]).astype(int)      # Convert coordinates to integers.Easier to work with integers
        for x, y, radius in circles:
            if (0 <= x < width and 0 <= y < height):       #To check if circle is in image
                if road_mask[y, x] > 0:     # Only consider circles located on the road
                    cv2.circle(obstacle_mask,(x, y),radius,255,-1)
                    print("Pothole detected at:",x,y)

    SAFETY_DISTANCE = 25
    obstacle_mask = cv2.dilate(
        obstacle_mask,
        np.ones(
            (SAFETY_DISTANCE, SAFETY_DISTANCE),
            np.uint8
        ),
        iterations=1
    )       #Enlarging obstacles to avoid any collision

    safe_area = road_mask.copy()
    safe_area[obstacle_mask > 0] = 0        # Remove obstacle regions by converting them to black(0)

    def heuristic(point1, point2):
        x1, y1 = point1
        x2, y2 = point2
        return math.sqrt((x2 - x1) ** 2 +(y2 - y1) ** 2)      #h(n). Return Eucleadina distance

    def astar(start, goal, grid):
        open_set = [(0, start)]
        came_from = {}
        cost = {start: 0}
        directions = [
            (-1, 0), (1, 0),
            (0, -1), (0, 1),
            (-1, -1), (-1, 1),
            (1, -1), (1, 1)
        ]

        while open_set:
            _, current = heapq.heappop(open_set)
            if current == goal:
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                path.reverse()
                return path
            for dx, dy in directions:
                nx = current[0] + dx
                ny = current[1] + dy
                if not (0 <= nx < grid.shape[1] and
                        0 <= ny < grid.shape[0]):
                    continue
                if grid[ny, nx] == 0:
                    continue
                neighbor = (nx, ny)
                new_cost = cost[current] + math.sqrt(dx**2 + dy**2)
                if neighbor not in cost or new_cost < cost[neighbor]:
                    cost[neighbor] = new_cost
                    came_from[neighbor] = current
                    priority = new_cost + heuristic(neighbor, goal)
                    heapq.heappush(open_set,(priority, neighbor))
        return None

    def nearest_safe_point(point):
        x, y = point
        if (0 <= x < width and 0 <= y < height and safe_area[y, x] > 0):
            return point
        for radius in range(1, 51):
            for dx in range(-radius,radius + 1):
                for dy in range(-radius,radius + 1):
                    nx = x + dx
                    ny = y + dy
                    if (0 <= nx < width and 0 <= ny < height):
                        if safe_area[ny, nx] > 0:
                            return (nx, ny)
        return None     #No safe point found

    safe_checkpoints = []
    for checkpoint in checkpoints[j]:
        safe_point = nearest_safe_point(checkpoint)
        if safe_point is not None:
            safe_checkpoints.append(safe_point)
        else:
            print("WARNING: No safe point found near",checkpoint)

    final_path = []
    for i in range(len(safe_checkpoints) - 1):
        start = safe_checkpoints[i]
        goal = safe_checkpoints[i + 1]
        print("Calculating:",start,"→",goal)
        path = astar(start,goal,safe_area)
        if path is None:
            print("No safe path found between",start,"and",goal)
        else:
            final_path.extend(path)

    for i in range(len(final_path) - 1):
        point1 = final_path[i]
        point2 = final_path[i + 1]
        cv2.line(output,point1,point2,(0, 255, 0),4)

    for point in safe_checkpoints[:-1]:
        cv2.circle(output,point,7,(255, 0, 0),-1)

    if len(safe_checkpoints) > 0:
        start = safe_checkpoints[0]
        cv2.circle(output,start,12,(0, 0, 255),-1)
        cv2.putText(output,"START",(start[0] + 15,start[1]),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0, 0, 255),2)

    display_width = 800
    display_height = 800
    display_image = cv2.resize(output,(display_width, display_height))
    cv2.imshow(f"Output - {j+1}",display_image)
    cv2.imwrite(f"Output/Output - {j+1}.jpeg",display_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()