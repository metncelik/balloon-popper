import cv2
import numpy as np

def detect_balloons(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)
    
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=50,
        param1=10,
        param2=25,
        minRadius=10,
        maxRadius=100
    )
    
    if circles is None:
        return []
    
    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 100, 100])
    upper_red2 = np.array([180, 255, 255])
    
    lower_blue = np.array([30, 100, 100])
    upper_blue = np.array([130, 255, 255])
    
    detected_balloons = []
    circles = np.uint16(np.around(circles))
    
    for circle in circles[0, :]:
        center = (circle[0], circle[1])
        radius = circle[2]
        
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        cv2.circle(mask, center, radius, 255, -1)
        
        masked_hsv = cv2.bitwise_and(hsv, hsv, mask=mask)
        
        roi = masked_hsv[max(0, circle[1]-radius):min(hsv.shape[0], circle[1]+radius),
                        max(0, circle[0]-radius):min(hsv.shape[1], circle[0]+radius)]
        
        red_pixels1 = cv2.inRange(roi, lower_red1, upper_red1)
        red_pixels2 = cv2.inRange(roi, lower_red2, upper_red2)
        red_count = cv2.countNonZero(red_pixels1) + cv2.countNonZero(red_pixels2)
        
        blue_pixels = cv2.inRange(roi, lower_blue, upper_blue)
        blue_count = cv2.countNonZero(blue_pixels)
        
        x = int(circle[0] - radius)
        y = int(circle[1] - radius)
        w = int(2 * radius)
        h = int(2 * radius)
        bbox = (x, y, w, h)
        
        if red_count > blue_count and red_count > 100:
            balloon_info = {
                'color': 'red',
                'bbox': bbox,
                'center': center,
                'radius': radius
            }
            detected_balloons.append(balloon_info)
        if blue_count > red_count and blue_count > 100:
            balloon_info = {
                'color': 'blue',
                'bbox': bbox,
                'center': center,
                'radius': radius
            }
            detected_balloons.append(balloon_info)
    
    return detected_balloons

def draw_balloon_detections(image, detections):
    result = image.copy()
    
    for balloon in detections:
        x, y, w, h = balloon['bbox']
        color = (0, 0, 255) if balloon['color'] == 'red' else (255, 0, 0)
        
        cv2.rectangle(result, (x, y), (x + w, y + h), color, 2)
        
        cv2.circle(result, balloon['center'], 2, color, -1)
    
    return result

if __name__ == "__main__":
    image_path = './test_frames/frame_0020.jpg'
    image = cv2.imread(image_path)
    
    if image is None:
        print(f"Error: Could not read image from {image_path}")
        exit(1)
    
    detections = detect_balloons(image)
    
    result = draw_balloon_detections(image, detections)
    
    cv2.imshow('Balloon Detection', result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    cv2.imwrite('result.jpg', result)
    
    print(f"Found {len(detections)} balloons:")
    for i, balloon in enumerate(detections, 1):
        print(f"Balloon {i}:")
        print(f"  Color: {balloon['color']}")
        print(f"  Position: {balloon['center']}")
        print(f"  Radius: {balloon['radius']}")
