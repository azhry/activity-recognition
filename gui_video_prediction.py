import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from ultralytics import YOLO
import tkinter as tk
from tkinter import filedialog, messagebox
import sys

# --- Configuration ---
YOLO_MODEL_PATH = "yolov8n.pt"
CLASSIFICATION_MODEL_PATH = "helmet_detection_model.h5"
LABEL_MAPPING_PATH = "image_label_mapping.npy"
IMG_SIZE = 128

def select_video_source():
    """
    Creates a UI dialog to let the user select between a webcam or a video file.
    Returns the video source (0 for webcam, or a file path) or None if canceled.
    """
    root = tk.Tk()
    root.withdraw()

    source_choice = None

    def use_webcam():
        nonlocal source_choice
        source_choice = 0
        root.destroy()

    def select_file():
        nonlocal source_choice
        file_path = filedialog.askopenfilename(
            title="Select a Video File",
            filetypes=[
                ("Video files", "*.mp4 *.avi *.mov *.mkv"),
                ("All files", "*.*")
            ]
        )
        if file_path:
            source_choice = file_path
        root.destroy()
    
    # Create the dialog window
    dialog_window = tk.Toplevel(root)
    dialog_window.title("Choose Video Source")
    
    # Center the dialog window
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    window_width = 300
    window_height = 100
    x_cordinate = int((screen_width / 2) - (window_width / 2))
    y_cordinate = int((screen_height / 2) - (window_height / 2))
    dialog_window.geometry(f"{window_width}x{window_height}+{x_cordinate}+{y_cordinate}")

    label = tk.Label(dialog_window, text="Please choose your video source:", padx=10, pady=10)
    label.pack()

    webcam_button = tk.Button(dialog_window, text="Use Webcam", command=use_webcam)
    webcam_button.pack(side=tk.LEFT, padx=10, pady=5)

    file_button = tk.Button(dialog_window, text="Select Video File", command=select_file)
    file_button.pack(side=tk.RIGHT, padx=10, pady=5)
    
    dialog_window.protocol("WM_DELETE_WINDOW", root.destroy)
    root.mainloop()

    return source_choice

# --- The rest of your main() function is below this line ---

def main():
    """
    Main function to run real-time predictions.
    """
    video_source = select_video_source()

    if video_source is None:
        print("No video source selected. Exiting.")
        sys.exit()

    # Load the models and label mapping
    try:
        yolo_model = YOLO(YOLO_MODEL_PATH)
        classification_model = load_model(CLASSIFICATION_MODEL_PATH)
        label_mapping = np.load(LABEL_MAPPING_PATH, allow_pickle=True)
        class_names = label_mapping.tolist()
        print(f"Loaded classification model with {len(class_names)} classes: {class_names}")
    except Exception as e:
        print(f"Error loading models or mapping files: {e}")
        sys.exit()

    # Open the video source
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print(f"Error: Could not open video source '{video_source}'.")
        sys.exit()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video or an error occurred.")
            break

        # Use YOLO to detect all people in the current frame
        yolo_results = yolo_model(frame, classes=0, verbose=False)

        # Process the results
        for result in yolo_results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                # Ensure bounding box coordinates are within frame bounds
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(frame.shape[1], x2)
                y2 = min(frame.shape[0], y2)

                # Crop the person's bounding box
                cropped_person = frame[y1:y2, x1:x2]
                
                if cropped_person.size == 0:
                    continue

                # Preprocess the cropped image for the classification model
                resized_image = cv2.resize(cropped_person, (IMG_SIZE, IMG_SIZE))
                normalized_image = resized_image / 255.0
                
                # The model expects a batch of images, so we add a new dimension
                input_data = np.expand_dims(normalized_image, axis=0)
                
                # Make a prediction
                prediction = classification_model.predict(input_data, verbose=0)[0]
                predicted_class_index = np.argmax(prediction)
                predicted_class_name = class_names[predicted_class_index]
                confidence = prediction[predicted_class_index]

                # Display the results
                label = f"{predicted_class_name}: {confidence:.2f}"
                color = (0, 255, 0) if "helmet" in predicted_class_name.lower() else (0, 0, 255)

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Display the output frame
        cv2.imshow('Real-time Prediction', frame)
        
        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Clean up
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()