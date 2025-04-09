import cv2
import time
import numpy as np

# Initialize video capture
video = cv2.VideoCapture('car-and-pedestrian-video0.mp4')

# Load classifiers
car_tracker_file = 'cars.xml'
pedestrian_tracker = 'haarcascade_fullbody.xml'
car_tracker = cv2.CascadeClassifier(car_tracker_file)
pedestrian_tracker = cv2.CascadeClassifier(pedestrian_tracker)

# Initialize counters and FPS variables
total_cars = 0
total_pedestrians = 0
frame_count = 0
start_time = time.time()
fps = 0

# Create a window with a specific size
cv2.namedWindow('Car and Pedestrian Tracker', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Car and Pedestrian Tracker', 1280, 720)

while True:
    # Reading the current frame
    (read_successful, frame) = video.read()

    if not read_successful:
        break

    # Convert to grayscale
    greyscaled_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect cars and pedestrians
    cars = car_tracker.detectMultiScale(greyscaled_frame)
    pedestrians = pedestrian_tracker.detectMultiScale(greyscaled_frame)

    # Update counters
    total_cars += len(cars)
    total_pedestrians += len(pedestrians)
    frame_count += 1

    # Calculate FPS
    if frame_count % 30 == 0:  # Update FPS every 30 frames
        end_time = time.time()
        fps = frame_count / (end_time - start_time)

    # Draw rectangles and labels for cars
    for (x, y, w, h) in cars:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
        cv2.putText(frame, 'Car', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # Draw rectangles and labels for pedestrians
    for (x, y, w, h) in pedestrians:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
        cv2.putText(frame, 'Pedestrian', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

    # Create a stats panel
    stats_panel = np.zeros((100, frame.shape[1], 3), dtype=np.uint8)
    
    # Add statistics to the panel
    cv2.putText(stats_panel, f'Cars Detected: {total_cars}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(stats_panel, f'Pedestrians Detected: {total_pedestrians}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(stats_panel, f'FPS: {fps:.2f}', (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Combine the stats panel with the frame
    combined_frame = np.vstack((stats_panel, frame))

    # Display the combined frame
    cv2.imshow('Car and Pedestrian Tracker', combined_frame)

    # Handle key presses
    key = cv2.waitKey(1)
    if key == ord('q') or key == ord('Q'):
        break
    elif key == ord('r') or key == ord('R'):  # Reset counters
        total_cars = 0
        total_pedestrians = 0
        frame_count = 0
        start_time = time.time()

# Release resources
video.release()
cv2.destroyAllWindows()

print("Code completed")
print(f"Total cars detected: {total_cars}")
print(f"Total pedestrians detected: {total_pedestrians}")
