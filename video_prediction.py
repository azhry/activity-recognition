import cv2
import numpy as np
import tensorflow as tf
from ultralytics import YOLO
import collections

# --- Configuration (must match model and preprocessing) ---
IMG_SIZE = 128
NUM_FRAMES = 30
MODEL_PATH = "best_model.h5"
LABEL_MAPPING_PATH = "label_mapping.npy"

# --- Main Prediction Function ---
def predict_activity(model, frame_queue, labels):
    """
    Predicts the activity from the sequence of frames in the queue.
    """
    if len(frame_queue) == NUM_FRAMES:
        sequence = np.array(frame_queue)
        sequence = np.expand_dims(sequence, axis=0) # Shape: (1, 30, 128, 128, 3)
        
        predictions = model.predict(sequence, verbose=0)
        predicted_class_index = np.argmax(predictions)
        predicted_activity = labels[predicted_class_index]
        confidence = np.max(predictions)
        
        return predicted_activity, confidence
        
    return "Collecting frames...", 0.0

def main(video_path, output_video_path):
    """
    Predicts activity for multiple people in a video file and saves the result.
    """
    print("Loading the trained model...")
    model = tf.keras.models.load_model(MODEL_PATH)
    labels = np.load(LABEL_MAPPING_PATH)
    print("Model and labels loaded successfully.")

    yolo_model = YOLO("yolov8n.pt")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video file.")
        return
        
    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    print(f"Video Info: {total_frames} frames @ {fps:.2f} FPS")
    print(f"Output video will be saved to: {output_video_path}")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Codec for MP4
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))
        
    # Dictionary to store each person's data (frame queue, last prediction)
    person_data = {}
    
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Use YOLO's tracking feature to detect people
        results = yolo_model.track(frame, classes=0, conf=0.5, persist=True, verbose=False)
        
        # A set to keep track of currently active person IDs
        current_person_ids = set()

        if results and results[0].boxes.id is not None:
            # Iterate through all detected and tracked persons
            for box in results[0].boxes:
                x1, y1, x2, y2 = box.xyxy[0].numpy().astype(int)
                track_id = int(box.id[0])
                current_person_ids.add(track_id)

                person_crop = frame[y1:y2, x1:x2]

                if person_crop.size != 0:
                    processed_person = cv2.resize(person_crop, (IMG_SIZE, IMG_SIZE))
                    processed_person = cv2.cvtColor(processed_person, cv2.COLOR_BGR2RGB)
                    processed_person = processed_person / 255.0

                    # Check if this person is new
                    if track_id not in person_data:
                        person_data[track_id] = {
                            "queue": collections.deque(maxlen=NUM_FRAMES),
                            "prediction": "Collecting...",
                            "confidence": 0.0
                        }
                    
                    person_data[track_id]["queue"].append(processed_person)

                    # Get the prediction for this person
                    pred, conf = predict_activity(model, person_data[track_id]["queue"], labels)
                    person_data[track_id]["prediction"] = pred
                    person_data[track_id]["confidence"] = conf
        
        # Draw all current predictions on the frame
        if results and results[0].boxes.id is not None:
            for box in results[0].boxes:
                x1, y1, x2, y2 = box.xyxy[0].numpy().astype(int)
                track_id = int(box.id[0])

                pred = person_data[track_id]["prediction"]
                conf = person_data[track_id]["confidence"]

                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(frame, f"ID: {track_id} | Activity: {pred}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                cv2.putText(frame, f"Conf: {conf:.2f}", (x1, y1 + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 1)

        # Write the processed frame to the output video
        out.write(frame)

        # Print progress
        frame_count += 1
        if frame_count % 50 == 0:
            progress = (frame_count / total_frames) * 100
            print(f"Progress: {progress:.2f}% complete...")

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print("Video processing and saving complete.")

if __name__ == '__main__':
    # Set the path to your input and output video files
    video_file = "sports_moments.f401.mp4"
    output_file = "prediction_output.mp4"
    main(video_file, output_file)