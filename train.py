import os
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint
from model import build_model
from data_generator import DataGenerator, get_data_paths_and_labels

# --- Configuration ---
DATA_DIR = "processed_data"
MODEL_SAVE_PATH = "best_model.h5"
NUM_CLASSES = 50
EPOCHS = 20
BATCH_SIZE = 32

def train_model():
    """
    Loads data paths, splits them, builds the model, and trains using a data generator.
    """
    # Get all file paths and their integer-encoded labels
    print("Collecting data paths and labels...")
    data_paths, y_encoded = get_data_paths_and_labels(DATA_DIR)

    # Split data paths into training and validation sets
    X_train_paths, X_val_paths, y_train, y_val = train_test_split(
        data_paths, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    print(f"Training on {len(X_train_paths)} samples, validating on {len(X_val_paths)} samples.")

    # Create the data generators
    train_generator = DataGenerator(X_train_paths, y_train, BATCH_SIZE)
    val_generator = DataGenerator(X_val_paths, y_val, BATCH_SIZE)

    # Build the model from model.py
    model = build_model()
    
    # Create a ModelCheckpoint callback to save the best model weights
    checkpoint = ModelCheckpoint(
        MODEL_SAVE_PATH,
        monitor='val_accuracy',
        verbose=1,
        save_best_only=True,
        mode='max'
    )

    # Start training using the generators
    print("Starting model training...")
    history = model.fit(
        train_generator,
        epochs=EPOCHS,
        validation_data=val_generator,
        callbacks=[checkpoint]
    )

    print("Training finished.")

if __name__ == "__main__":
    train_model()