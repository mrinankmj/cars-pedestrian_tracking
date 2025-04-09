import cv2
import time
import numpy as np

# Initialize video capture
video = cv2.VideoCapture('car-and-pedestrian-video0.mp4')

# Get video properties
total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
video_fps = int(video.get(cv2.CAP_PROP_FPS))
video_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
video_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Load classifiers
car_tracker_file = 'cars.xml'
pedestrian_tracker = 'haarcascade_fullbody.xml'
car_tracker = cv2.CascadeClassifier(car_tracker_file)
pedestrian_tracker = cv2.CascadeClassifier(pedestrian_tracker)

# Initialize variables
frame_count = 0
start_time = time.time()
fps = 0
show_help = True  # Toggle for help overlay

# Create a window with a specific size
cv2.namedWindow('Car and Pedestrian Tracker', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Car and Pedestrian Tracker', 1280, 720)

def create_help_overlay(frame_width):
    help_panel = np.zeros((100, frame_width, 3), dtype=np.uint8)
    cv2.putText(help_panel, "Controls:", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.putText(help_panel, "Q: Quit | H: Toggle Help | P: Pause/Resume", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    return help_panel

def create_progress_bar(frame_width, progress):
    # Create progress bar panel
    progress_height = 20
    progress_panel = np.zeros((progress_height, frame_width, 3), dtype=np.uint8)
    
    # Draw progress bar background
    cv2.rectangle(progress_panel, (0, 0), (frame_width, progress_height), (50, 50, 50), -1)
    
    # Calculate progress bar width
    progress_width = int(frame_width * progress)
    
    # Draw progress bar
    cv2.rectangle(progress_panel, (0, 0), (progress_width, progress_height), (0, 255, 0), -1)
    
    # Add time information
    current_time = frame_count / video_fps
    total_time = total_frames / video_fps
    time_text = f"{int(current_time)}s / {int(total_time)}s"
    cv2.putText(progress_panel, time_text, (10, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return progress_panel

paused = False
while True:
    if not paused:
        # Reading the current frame
        (read_successful, frame) = video.read()

        if not read_successful:
            break

        # Convert to grayscale
        greyscaled_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect cars and pedestrians
        cars = car_tracker.detectMultiScale(greyscaled_frame)
        pedestrians = pedestrian_tracker.detectMultiScale(greyscaled_frame)

        # Update frame count and calculate FPS
        frame_count += 1
        if frame_count % 30 == 0:
            current_time = time.time()
            fps = 30 / (current_time - start_time)
            start_time = current_time

        # Draw rectangles and labels for cars
        for (x, y, w, h) in cars:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
            cv2.putText(frame, 'Car', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # Draw rectangles and labels for pedestrians
        for (x, y, w, h) in pedestrians:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
            cv2.putText(frame, 'Pedestrian', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        # Create a stats panel
        stats_panel = np.zeros((80, frame.shape[1], 3), dtype=np.uint8)
        
        # Add statistics to the panel
        cv2.putText(stats_panel, f'FPS: {fps:.1f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(stats_panel, f'Frame: {frame_count}/{total_frames}', (200, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(stats_panel, f'Cars: {len(cars)}  Pedestrians: {len(pedestrians)}', (500, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Add status line
        status = "PLAYING" if not paused else "PAUSED"
        cv2.putText(stats_panel, status, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0) if not paused else (0, 0, 255), 2)

        # Create progress bar
        progress = frame_count / total_frames
        progress_bar = create_progress_bar(frame.shape[1], progress)

        # Combine panels with the frame
        combined_frame = np.vstack((stats_panel, frame, progress_bar))
        
        # Add help overlay if enabled
        if show_help:
            help_overlay = create_help_overlay(frame.shape[1])
            combined_frame = np.vstack((help_overlay, combined_frame))

    # Display the combined frame
    cv2.imshow('Car and Pedestrian Tracker', combined_frame)

    # Handle key presses
    key = cv2.waitKey(1)
    if key == ord('q') or key == ord('Q'):
        break
    elif key == ord('h') or key == ord('H'):
        show_help = not show_help
    elif key == ord('p') or key == ord('P'):
        paused = not paused

# Release resources
video.release()
cv2.destroyAllWindows()

print("Code completed")
