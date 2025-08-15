# Activity Recognition using 3D CNN and YOLOv8

This project implements a system for recognizing human activities from video files. It uses a 3D Convolutional Neural Network (3D CNN) to learn spatiotemporal features and integrates with the YOLOv8 model for multi-person tracking. The system is trained on the UCF101 dataset.

## Features

- **3D CNN Model:** A custom model built with TensorFlow/Keras.
- **YOLOv8 Integration:** Used for efficient person detection and tracking.
- **Multi-Person Support:** Capable of tracking multiple individuals simultaneously.
- **Data Preprocessing Pipeline:** A script to prepare the UCF101 dataset for training.
- **Video Prediction Script:** An application to predict activities in new video files.


## Getting Started

### Prerequisites

- Python 3.8+
- Git

### Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/azhry/activity-recognition.git
    cd activity-recognition
    ```
2.  Create and activate a virtual environment:
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```
3.  Install the required packages:
    Create a `requirements.txt` file with the contents below, then install:
    ```bash
    pip install -r requirements.txt
    ```

### `requirements.txt`
```
tensorflow
ultralytics
opencv-python
scikit-learn
tqdm
numpy
```

### Data Preparation

1.  Download the UCF101 dataset and place it in the project root.
2.  Run the preprocessing script to prepare the data for training.
    ```bash
    python3 data_preprocessing.py
    ```

## Usage

### 1. Train the Model

Run this script to train the 3D CNN model on the processed data.
```bash
python3 train.py
```
### 2. Run Predictions
Update the video_file path in the script, then run it to get predictions.

```bash
python3 video_prediction.py
```
