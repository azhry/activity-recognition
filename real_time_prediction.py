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
    # Ensure the queue is full before making a prediction
    if len(frame_queue) == NUM_FRAMES:
        # Convert the frame queue to a numpy array for the model
        sequence = np.array(frame_queue)
        
        # Add a batch dimension (1, NUM_FRAMES, IMG_SIZE, IMG_SIZE, 3)
        sequence = np.expand_dims(sequence, axis=0)
        
        # Predict the activity
        predictions = model.predict(sequence, verbose=0)
        
        # Get the predicted class index
        predicted_class_index = np.argmax(predictions)
        
        # Get the corresponding activity label
        predicted_activity = labels[predicted_class_index]
        
        return predicted_activity, predictions[0][predicted_class_index]
        
    return "Collecting frames...", 0.0

def main():
    """
    Runs the real-time activity recognition system for multiple people.
    """
    # Load the trained model and labels
    print("Loading the trained model...")
    model = tf.keras.models.load_model(MODEL_PATH)
    labels = np.load(LABEL_MAPPING_PATH)
    print("Model and labels loaded successfully.")

    # Initialize YOLO for person detection and tracking
    yolo_model = YOLO("yolov8n.pt")
    
    # Initialize webcam capture
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open video stream or file.")
        return
        
    # Dictionary to store a frame queue and prediction for each person
    # The key is the track_id from YOLO
    person_data = {}
    
    print("Starting real-time activity recognition...")
    print("Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)

        # Use YOLO's tracking feature to detect people
        # The 'persist=True' parameter helps maintain the track_ids
        results = yolo_model.track(frame, classes=0, conf=0.5, persist=True, verbose=False)
        
        if results and results[0].boxes.id is not None:
            # Iterate through all detected and tracked persons
            for box in results[0].boxes:
                x1, y1, x2, y2 = box.xyxy[0].numpy().astype(int)
                track_id = int(box.id[0])

                # Get the cropped person region
                person_crop = frame[y1:y2, x1:x2]

                if person_crop.size != 0:
                    # Resize and normalize the cropped frame
                    processed_person = cv2.resize(person_crop, (IMG_SIZE, IMG_SIZE))
                    processed_person = cv2.cvtColor(processed_person, cv2.COLOR_BGR2RGB)
                    processed_person = processed_person / 255.0

                    # Check if this person is new or already tracked
                    if track_id not in person_data:
                        person_data[track_id] = {
                            "queue": collections.deque(maxlen=NUM_FRAMES),
                            "prediction": "Collecting frames...",
                            "confidence": 0.0
                        }
                    
                    # Add the frame to this person's queue
                    person_data[track_id]["queue"].append(processed_person)

                    # Get the prediction for this person
                    pred, conf = predict_activity(model, person_data[track_id]["queue"], labels)
                    person_data[track_id]["prediction"] = pred
                    person_data[track_id]["confidence"] = conf
                    
                    # Draw the bounding box and prediction on the frame
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    cv2.putText(frame, f"ID: {track_id} | Activity: {pred}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                    cv2.putText(frame, f"Conf: {conf:.2f}", (x1, y1 + 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 1)
        
        # Show the output frame
        cv2.imshow("Real-Time Activity Recognition (Multi-Person)", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Clean up
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()