import cv2
import numpy as np
from utils import extract_frames
from detect_balloons import detect_balloons, draw_balloon_detections

def track_balloons(frames):
    balloon_positions = []
    
    for frame in frames:
        detections = detect_balloons(frame)
        if detections:
            centers = np.array([d['center'] for d in detections])
            if len(centers) > 0:
                hull = cv2.convexHull(centers.astype(np.float32))
                hull_points = hull.reshape(-1, 2)
                
                if len(hull_points) >= 2:
                    balloon_positions.append(hull_points)
                elif len(hull_points) == 1:
                    balloon_positions.append(hull_points[0])
    
    if not balloon_positions:
        return None, frames
    
    all_points = np.vstack(balloon_positions)
    
    if len(all_points) >= 3:
        final_hull = cv2.convexHull(all_points.astype(np.float32))
        path = final_hull.reshape(-1, 2)
    else:
        path = all_points
    
    path = np.array(path, dtype=np.int32)
    if len(path) > 0:
        path = np.vstack((path, path[0]))
    
    return path, frames

def draw_path(frame, path, thickness=2):
    result = frame.copy()
    
    if len(path) >= 2:
        cv2.polylines(result, [path], True, (255, 255, 255), thickness, lineType=cv2.LINE_AA)
        
        num_arrows = 8
        path_length = len(path)
        for i in range(num_arrows):
            idx = (i * path_length) // num_arrows
            next_idx = ((idx + path_length//num_arrows//2) % path_length)
            
            pt1 = tuple(path[idx])
            pt2 = tuple(path[next_idx])
            
            cv2.arrowedLine(result, pt1, pt2, (255, 255, 255), 
                           thickness, tipLength=0.2)
    
    return result

def predict_path(video_path):
    frames = extract_frames(video_path, frame_interval=15)
    path, frames = track_balloons(frames)
    
    if path is not None and len(frames) > 0:
        blended_frame = frames[0].astype(float)
        for frame in frames[1:]:
            blended_frame = cv2.addWeighted(blended_frame, 0.7, frame.astype(float), 0.3, 0)
        
        blended_frame = np.uint8(blended_frame)
        
        result = draw_path(blended_frame, path)
        
        detections = detect_balloons(frames[-1])
        result = draw_balloon_detections(result, detections)
        
        cv2.imshow('Predicted Path', result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        cv2.imwrite('predicted_path.jpg', result)

if __name__ == "__main__":
    predict_path("./test_videos/balloons_test_2.mp4")