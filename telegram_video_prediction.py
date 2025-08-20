import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from ultralytics import YOLO
import tkinter as tk
from tkinter import filedialog, messagebox
import sys
import telegram
import asyncio
import time
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# --- Configuration ---
YOLO_MODEL_PATH = "yolov8n.pt"
CLASSIFICATION_MODEL_PATH = "helmet_detection_model.h5"
LABEL_MAPPING_PATH = "image_label_mapping.npy"
IMG_SIZE = 128

# --- Telegram Configuration ---
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

# Initialize a global bot object for Telegram
telegram_bot = telegram.Bot(token=TELEGRAM_BOT_TOKEN)

# To prevent sending too many notifications, we'll add a simple rate limit
last_notification_time = 0
NOTIFICATION_INTERVAL_SECONDS = int(os.getenv("NOTIFICATION_INTERVAL_SECONDS"))
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD"))

print(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, NOTIFICATION_INTERVAL_SECONDS)

async def send_telegram_notification(image, message):
    """
    Sends a message and a screenshot of the detected violation to a Telegram group.
    """
    try:
        # Encode the image to a format Telegram can handle
        _, img_encoded = cv2.imencode('.jpg', image)
        
        # Send the photo and a caption asynchronously
        await telegram_bot.send_photo(
            chat_id=TELEGRAM_CHAT_ID,
            photo=img_encoded.tobytes(),
            caption=message
        )
        print("Telegram notification sent successfully.")
    except Exception as e:
        print(f"Error sending Telegram notification: {e}")

def select_video_source():
    # This function is unchanged
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
    
    dialog_window = tk.Toplevel(root)
    dialog_window.title("Choose Video Source")
    
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

async def main_async():
    """
    The main asynchronous function that runs the video processing loop.
    """
    video_source = select_video_source()

    if video_source is None:
        print("No video source selected. Exiting.")
        sys.exit()

    try:
        yolo_model = YOLO(YOLO_MODEL_PATH)
        classification_model = load_model(CLASSIFICATION_MODEL_PATH)
        label_mapping = np.load(LABEL_MAPPING_PATH, allow_pickle=True)
        class_names = label_mapping.tolist()
        print(f"Loaded classification model with {len(class_names)} classes: {class_names}")
    except Exception as e:
        print(f"Error loading models or mapping files: {e}")
        sys.exit()

    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print(f"Error: Could not open video source '{video_source}'.")
        sys.exit()
    
    global last_notification_time

    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video or an error occurred.")
            break

        yolo_results = yolo_model(frame, classes=0, verbose=False)

        person_without_helmet_detected = False

        for result in yolo_results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(frame.shape[1], x2)
                y2 = min(frame.shape[0], y2)

                cropped_person = frame[y1:y2, x1:x2]
                
                if cropped_person.size == 0:
                    continue

                resized_image = cv2.resize(cropped_person, (IMG_SIZE, IMG_SIZE))
                normalized_image = resized_image / 255.0
                input_data = np.expand_dims(normalized_image, axis=0)
                
                prediction = classification_model.predict(input_data, verbose=0)[0]
                predicted_class_index = np.argmax(prediction)
                predicted_class_name = class_names[predicted_class_index]
                confidence = prediction[predicted_class_index]
                
                label = f"{predicted_class_name}: {confidence:.2f}"
                color = (0, 255, 0) if predicted_class_name.lower() in ['helmet', 'hat'] else (0, 0, 255)

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                if predicted_class_name.lower() not in ['helmet', 'hat'] and confidence > CONFIDENCE_THRESHOLD:
                    person_without_helmet_detected = True

        current_time = time.time()
        
        if person_without_helmet_detected and (current_time - last_notification_time) > NOTIFICATION_INTERVAL_SECONDS:
            message = "⚠️ WARNING: Person detected without a helmet!"
            asyncio.create_task(send_telegram_notification(frame, message))
            last_notification_time = current_time

        cv2.imshow('Real-time Prediction', frame)
        
        # This small sleep yields control back to the asyncio event loop
        # so it can process the pending Telegram notification task.
        await asyncio.sleep(0.01)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    try:
        asyncio.run(main_async())
    except RuntimeError as e:
        if "Event loop is closed" in str(e):
            print("Event loop was closed, but the application is exiting.")
        else:
            raise e