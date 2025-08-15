import os
import cv2
import numpy as np
from tqdm import tqdm # for a progress bar

# --- Configuration ---
DATA_DIR = "UCF50" # Your dataset folder
OUTPUT_DIR = "processed_data" # Where to save the processed data
IMG_SIZE = 128 # The size to resize our frames to
NUM_FRAMES = 30 # The number of frames to sample from each video

def create_directory(path):
    """Create a directory if it doesn't exist."""
    if not os.path.exists(path):
        os.makedirs(path)

def process_video(video_path, output_folder):
    """
    Reads a video, resizes the frames, and saves them as a numpy array.
    """
    cap = cv2.VideoCapture(video_path)
    frames = []

    # Get the total number of frames in the video
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Calculate step size to get NUM_FRAMES evenly distributed
    step = total_frames // NUM_FRAMES
    if step == 0:
        step = 1

    for i in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break

        # Only process frames at the calculated step
        if i % step == 0 and len(frames) < NUM_FRAMES:
            # Resize frame to our IMG_SIZE
            resized_frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
            # Convert to RGB (OpenCV reads as BGR)
            rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
            frames.append(rgb_frame)

    cap.release()
    
    # Pad or truncate the frames to ensure a consistent number of frames
    while len(frames) < NUM_FRAMES:
        frames.append(np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8))
        
    if len(frames) > NUM_FRAMES:
        frames = frames[:NUM_FRAMES]

    return np.array(frames)

def preprocess_dataset():
    """
    Iterates through the UCF101 dataset, processes each video, and saves
    the processed frames to the output directory.
    """
    create_directory(OUTPUT_DIR)
    
    class_names = sorted(os.listdir(DATA_DIR))
    class_names = [cls for cls in class_names if not cls.startswith('.')] # Ignore hidden files

    for class_name in tqdm(class_names, desc="Processing videos"):
        class_path = os.path.join(DATA_DIR, class_name)
        if not os.path.isdir(class_path):
            continue

        output_class_path = os.path.join(OUTPUT_DIR, class_name)
        create_directory(output_class_path)

        for video_file in os.listdir(class_path):
            if video_file.endswith(".avi"):
                video_path = os.path.join(class_path, video_file)
                
                # Process the video
                processed_frames = process_video(video_path, output_class_path)
                
                # Save the frames as a numpy file
                output_file_name = video_file.replace(".avi", ".npy")
                output_path = os.path.join(output_class_path, output_file_name)
                np.save(output_path, processed_frames)

    print("Data pre-processing complete!")

if __name__ == "__main__":
    preprocess_dataset()