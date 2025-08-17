import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks
from sklearn.model_selection import train_test_split
import numpy as np
import os

# --- Configuration ---
PROCESSED_DATA_DIR = "./processed_images"
IMG_SIZE = 128
EPOCHS = 30
BATCH_SIZE = 32

def create_model(num_classes):
    """
    Builds the 2D Convolutional Neural Network (CNN) model.
    """
    model = models.Sequential([
        # Input layer expects a single image (IMG_SIZE, IMG_SIZE, 3)
        layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3)),
        
        # 2D Convolutional Layers
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        # Flatten the output for the Dense layers
        layers.Flatten(),
        
        # Dense layers for classification
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        
        # Final output layer with NUM_CLASSES neurons
        layers.Dense(num_classes, activation='softmax')
    ])
    
    # Compile the model
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def get_data_paths_and_labels(data_dir):
    """
    Collects paths to all processed image files and their corresponding labels.
    """
    data_paths = []
    labels = []
    
    # Load the label mapping to know the classes
    label_encoder_classes = np.load('image_label_mapping.npy')
    label_mapping = {name: i for i, name in enumerate(label_encoder_classes)}

    class_names = os.listdir(data_dir)
    
    for class_name in class_names:
        class_dir = os.path.join(data_dir, class_name)
        if not os.path.isdir(class_dir):
            continue
        
        for file_name in os.listdir(class_dir):
            if file_name.endswith('.npy'):
                data_paths.append(os.path.join(class_dir, file_name))
                labels.append(label_mapping[class_name])
                
    return data_paths, labels, len(label_encoder_classes)

def data_generator(data_paths, batch_size):
    """
    A generator that loads processed image data in batches.
    """
    while True:
        # Shuffle the data
        indices = np.random.permutation(len(data_paths))
        data_paths = [data_paths[i] for i in indices]
        
        for i in range(0, len(data_paths), batch_size):
            batch_paths = data_paths[i:i + batch_size]
            
            # Load the data from the .npy files
            batch_images = []
            batch_labels = []
            for path in batch_paths:
                try:
                    data = np.load(path, allow_pickle=True).item()
                    batch_images.append(data['image'])
                    batch_labels.append(data['label'])
                except Exception as e:
                    print(f"Warning: Skipping file {path} due to error: {e}")
                    continue
            
            if batch_images:
                yield np.array(batch_images), np.array(batch_labels)

def train_model():
    """
    Main function to prepare data and train the model.
    """
    print("Collecting data paths and labels...")
    X_paths, y, num_classes = get_data_paths_and_labels(PROCESSED_DATA_DIR)
    
    if len(X_paths) == 0:
        print("Error: No processed data found. Please run data_preprocessing_voc.py first.")
        return
        
    print(f"Found {len(X_paths)} processed images across {num_classes} classes.")
    
    # Split the data into training and validation sets
    X_train_paths, X_val_paths, y_train, y_val = train_test_split(
        X_paths, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training on {len(X_train_paths)} samples, validating on {len(X_val_paths)} samples.")
    
    # Create the model
    model = create_model(num_classes)
    model.summary()
    
    # Create data generators
    train_generator = data_generator(X_train_paths, BATCH_SIZE)
    val_generator = data_generator(X_val_paths, BATCH_SIZE)
    
    # Define callbacks
    checkpoint_callback = callbacks.ModelCheckpoint(
        'helmet_detection_model.h5', save_best_only=True, monitor='val_loss'
    )
    early_stopping_callback = callbacks.EarlyStopping(
        monitor='val_loss', patience=10, restore_best_weights=True
    )
    
    # Train the model
    history = model.fit(
        train_generator,
        steps_per_epoch=len(X_train_paths) // BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=val_generator,
        validation_steps=len(X_val_paths) // BATCH_SIZE,
        callbacks=[checkpoint_callback, early_stopping_callback]
    )
    
    print("\nTraining complete! The best model is saved as 'helmet_detection_model.h5'")

if __name__ == "__main__":
    train_model()