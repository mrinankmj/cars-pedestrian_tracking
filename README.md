# Car and Pedestrian Tracking System

A real-time object detection system that tracks cars and pedestrians in video streams using OpenCV and Haar Cascade classifiers.

![image](https://github.com/user-attachments/assets/481f162f-ec04-4d84-b234-4e8c74b4ca65)

## Features

- 🚗 Real-time car detection
- 🚶 Real-time pedestrian detection
- 📊 FPS counter
- ⏱️ Progress bar with time information
- 📈 Real-time object counting
- 🎥 Video playback controls
- 🖥️ Enhanced UI with statistics overlay

## Requirements

- Python 3.x
- OpenCV (cv2)
- NumPy

## Installation

1. Clone the repository:
```bash
git clone https://github.com/mrinankmj/cars-pedestrian_tracking.git
cd cars-pedestrian_tracking
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install required packages:
```bash
pip install opencv-python numpy
```

## Usage

1. Run the detection program:
```bash
python dectecting_people_cars.py
```

2. Controls:
- `Q` or `q`: Quit the program
- `P` or `p`: Pause/Resume video
- `H` or `h`: Toggle help overlay

## UI Features

- **Statistics Panel**:
  - Current FPS
  - Frame count
  - Number of detected objects
  - Playback status

- **Progress Bar**:
  - Visual progress indicator
  - Current time / Total duration
  - Color-coded progress

- **Detection Visualization**:
  - Red boxes: Cars
  - Yellow boxes: Pedestrians
  - Object labels

## Project Structure

```
cars-pedestrian_tracking/
├── dectecting_people_cars.py  # Main detection script
├── cars.xml                   # Car detection classifier
├── haarcascade_fullbody.xml   # Pedestrian detection classifier
├── car-and-pedestrian-video0.mp4  # Sample video
└── README.md                  # This file
```

## How It Works

1. The program uses Haar Cascade classifiers to detect cars and pedestrians
2. Each frame is processed to identify objects
3. Detected objects are highlighted with bounding boxes
4. Real-time statistics are displayed
5. Progress bar shows video advancement

## Contributing

Feel free to submit issues and enhancement requests!

## License

This project is open source and available under the MIT License.

## Acknowledgments

- OpenCV for the computer vision framework
- Haar Cascade classifiers for object detection
