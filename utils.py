import cv2
import numpy as np
from pathlib import Path

def extract_frames(video_path, output_dir=None, frame_interval=1):
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    
    frames = []
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        if frame_count % frame_interval == 0:
            frames.append(frame)
            
            if output_dir:
                frame_path = output_dir / f"frame_{frame_count:04d}.jpg"
                cv2.imwrite(str(frame_path), frame)
        
        frame_count += 1
    
    cap.release()
    
    print(f"Extracted {len(frames)} frames from video")
    return frames

def get_video_info(video_path):
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    cap.release()
    
    return {
        'fps': fps,
        'frame_count': frame_count,
        'duration': duration,
        'width': width,
        'height': height
    }

if __name__ == "__main__":
    video_path = "test_videos/balloons_test_1.mp4" 
    output_dir = "extracted_frames"
    
    video_info = get_video_info(video_path)
    print("Video Information:")
    print(f"FPS: {video_info['fps']}")
    print(f"Frame Count: {video_info['frame_count']}")
    print(f"Duration: {video_info['duration']:.2f} seconds")
    print(f"Resolution: {video_info['width']}x{video_info['height']}")
    
    frames = extract_frames(video_path, output_dir, frame_interval=5)
