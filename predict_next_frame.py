import cv2
import numpy as np
from predict_path import track_balloons
from detect_balloons import detect_balloons
from utils import extract_frames

def find_closest_path_point(point, path):
    distances = np.sqrt(np.sum((path - point) ** 2, axis=1))
    return np.argmin(distances)

def predict_next_frame(frame, path, speed=1):
    detections = detect_balloons(frame)
    
    if not detections or path is None:
        return frame.copy()
    
    next_frame = np.zeros_like(frame)

    for balloon in detections:
        current_pos = np.array(balloon['center'])
        current_idx = find_closest_path_point(current_pos, path)
        
        next_idx = (current_idx + speed) % (len(path) - 1)
        next_pos = path[next_idx]
        
        color = (0, 0, 255) if balloon['color'] == 'red' else (255, 0, 0)
        cv2.circle(next_frame, 
                  tuple(map(int, next_pos)), 
                  balloon['radius'], 
                  color, 
                  -1, 
                  cv2.LINE_AA)
        
        highlight_pos = (int(next_pos[0] - balloon['radius']//2), 
                        int(next_pos[1] - balloon['radius']//2))
        highlight_size = balloon['radius'] // 3
        cv2.circle(next_frame, 
                  highlight_pos, 
                  highlight_size, 
                  (255, 255, 255), 
                  -1, 
                  cv2.LINE_AA)

    return next_frame

if __name__ == "__main__":
    video_path = "./test_videos/balloons_test_1.mp4"
    frame_interval = 10
    frames = extract_frames(video_path, output_dir="./test_frames", frame_interval=frame_interval)
    path, frames = track_balloons(frames)
    
    if path is not None and len(frames) > 0:
        current_frame = frames[0]
        
        cv2.namedWindow('Prediction', cv2.WINDOW_NORMAL)
        
        while True:
            cv2.imshow('Prediction', current_frame)
            
            next_frame = predict_next_frame(current_frame, path)
            
            key = cv2.waitKey(100)
            if key == ord('q'):
                break
            
            current_frame = next_frame
        
        cv2.destroyAllWindows()
