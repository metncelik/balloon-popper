import cv2
import numpy as np
from pathlib import Path

def create_random_loop_path(frame_size, num_points=10):
    """Create a random smooth loop path"""
    angles = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
    
    center_x = frame_size[0] // 2
    center_y = frame_size[1] // 2
    
    base_radius = min(frame_size) // 4
    radii = base_radius + np.random.randint(-100, 100, num_points)
    
    x_points = center_x + radii * np.cos(angles)
    y_points = center_y + radii * np.sin(angles)
    
    x_points = np.append(x_points, x_points[0])
    y_points = np.append(y_points, y_points[0])
    
    t = np.linspace(0, 1, len(x_points))
    t_smooth = np.linspace(0, 1, 300)
    
    x_smooth = np.interp(t_smooth, t, x_points)
    y_smooth = np.interp(t_smooth, t, y_points)
    
    window = 5
    x_smooth = np.convolve(x_smooth, np.ones(window)/window, mode='valid')
    y_smooth = np.convolve(y_smooth, np.ones(window)/window, mode='valid')
    
    x_smooth = np.clip(x_smooth, 100, frame_size[0]-100)
    y_smooth = np.clip(y_smooth, 100, frame_size[1]-100)
    
    return np.column_stack((x_smooth, y_smooth))

def get_safe_offsets(path_length, num_balloons, min_distance):
    max_attempts = 100
    for _ in range(max_attempts):
        offsets = []
        available_positions = set(range(path_length))
        
        for _ in range(num_balloons):
            if not available_positions:
                break
                
            new_offset = np.random.choice(list(available_positions))
            offsets.append(new_offset)
            
            for i in range(-min_distance, min_distance + 1):
                pos = (new_offset + i) % path_length
                available_positions.discard(pos)
        
        if len(offsets) == num_balloons:
            return offsets
    
    return [i * (path_length // num_balloons) for i in range(num_balloons)]

def add_lighting_effects(frame, center, radius, color):
    highlight_positions = [
        (center[0] - radius//2, center[1] - radius//2),  # Top-left highlight
        (center[0] + radius//3, center[1] - radius//3),  # Top-right smaller highlight
    ]
    
    highlight_sizes = [
        radius // 3,  # Main highlight
        radius // 4,  # Secondary highlight
    ]
    
    for pos, size in zip(highlight_positions, highlight_sizes):
        cv2.circle(frame, 
                  (int(pos[0]), int(pos[1])), 
                  size, 
                  (255, 255, 255), 
                  -1)
        cv2.circle(frame, 
                  (int(pos[0]), int(pos[1])), 
                  size, 
                  color, 
                  -1, 
                  cv2.LINE_AA)
        cv2.circle(frame, 
                  (int(pos[0]), int(pos[1])), 
                  size, 
                  (255, 255, 255), 
                  -1, 
                  cv2.LINE_AA)
    
    shadow_positions = [
        (center[0] + radius//2, center[1] + radius//2),  # Bottom-right shadow
    ]
    
    shadow_sizes = [
        radius // 2,  # Main shadow
    ]
    
    for pos, size in zip(shadow_positions, shadow_sizes):
        shadow_color = tuple(max(0, c - 100) for c in color)  # Darker version of balloon color
        cv2.circle(frame, 
                  (int(pos[0]), int(pos[1])), 
                  size, 
                  shadow_color, 
                  -1, 
                  cv2.LINE_AA)

def create_balloon_video(output_path, num_balloons=3, duration_seconds=10):
    frame_size = (1280, 720)
    fps = 30
    total_frames = duration_seconds * fps
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, frame_size)
    
    path = create_random_loop_path(frame_size)
    
    max_radius = 50
    min_distance = int(len(path) * (2 * max_radius) / min(frame_size))  # Scale based on path length
    
    safe_offsets = get_safe_offsets(len(path), num_balloons, min_distance)
    
    balloons = []
    for offset in safe_offsets:
        balloon = {
            'color': np.random.choice(['red', 'blue']),
            'radius': np.random.randint(30, 50),
            'offset': offset,
        }
        balloons.append(balloon)
    
    for frame_idx in range(total_frames):
        frame = np.zeros((frame_size[1], frame_size[0], 3), dtype=np.uint8)
        frame.fill(0)  # Black background
        
        for balloon in balloons:
            pos_idx = (frame_idx + balloon['offset']) % len(path)
            center = tuple(map(int, path[pos_idx]))
            
            if balloon['color'] == 'red':
                color = (0, 0, 255)  # BGR format
            else:
                color = (255, 0, 0)  # BGR format
            
            cv2.circle(frame, center, balloon['radius'], color, -1, cv2.LINE_AA)
            
            add_lighting_effects(frame, center, balloon['radius'], color)
        
        out.write(frame)
        
        if frame_idx % fps == 0:
            print(f"Processing frame {frame_idx}/{total_frames}")
    
    out.release()
    print(f"Video saved to {output_path}")

if __name__ == "__main__":
    output_dir = Path("test_videos")
    output_dir.mkdir(exist_ok=True)
    
    for i in range(5):
        output_path = str(output_dir / f"balloons_test_{i+1}.mp4")
        num_balloons = np.random.randint(2, 6)  # Random number of balloons (2-5)
        duration = np.random.randint(5, 11)  # Random duration (5-10 seconds)
        create_balloon_video(output_path, num_balloons, duration)
