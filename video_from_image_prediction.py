import cv2
import numpy as np
from tensorflow.keras.models import load_model # type: ignore
from ultralytics import YOLO

# --- Configuration ---
# Set the video source. You can use a file path or a webcam index (0 for default webcam)
VIDEO_SOURCE = 0 # Replace with your video file path or 0 for webcam
YOLO_MODEL_PATH = "yolov8n.pt"
CLASSIFICATION_MODEL_PATH = "helmet_detection_model.h5"
LABEL_MAPPING_PATH = "image_label_mapping.npy"
IMG_SIZE = 128

def main():
    """
    Main function to run real-time predictions on video.
    """
    # Load the models and label mapping
    try:
        yolo_model = YOLO(YOLO_MODEL_PATH)
        classification_model = load_model(CLASSIFICATION_MODEL_PATH)
        label_mapping = np.load(LABEL_MAPPING_PATH, allow_pickle=True)
        class_names = label_mapping.tolist()
        print(f"Loaded classification model with {len(class_names)} classes: {class_names}")
    except Exception as e:
        print(f"Error loading models or mapping files: {e}")
        return

    # Open the video source
    cap = cv2.VideoCapture(VIDEO_SOURCE)
    if not cap.isOpened():
        print(f"Error: Could not open video source '{VIDEO_SOURCE}'.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video or an error occurred.")
            break

        # Use YOLO to detect all people in the current frame
        # The 'classes=0' argument filters for the 'person' class only
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
                color = (0, 255, 0) if "helmet" in predicted_class_name.lower() or "hat" in predicted_class_name.lower() else (0, 0, 255)

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