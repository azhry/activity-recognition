import numpy as np
import os
import random
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf

class DataGenerator(tf.keras.utils.Sequence):
    """
    A data generator that loads batches of pre-processed video data from disk.
    This avoids loading the entire dataset into memory.
    """
    def __init__(self, data_paths, labels, batch_size=32, shuffle=True):
        self.data_paths = data_paths
        self.labels = labels
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        """Returns the number of batches per epoch."""
        return int(np.floor(len(self.data_paths) / self.batch_size))

    def __getitem__(self, index):
        """Generates one batch of data."""
        # Get the indices of the current batch
        indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]

        # Get the file paths and labels for this batch
        batch_paths = [self.data_paths[i] for i in indices]
        batch_labels = [self.labels[i] for i in indices]

        # Generate data for the batch
        X, y = self.__data_generation(batch_paths, batch_labels)
        return X, y

    def on_epoch_end(self):
        """Shuffle the indices at the end of each epoch."""
        self.indices = np.arange(len(self.data_paths))
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __data_generation(self, batch_paths, batch_labels):
        """Loads and processes a single batch of data."""
        X_batch = []
        y_batch = []
        
        # Get the input shape from the first element
        sample_frames = np.load(batch_paths[0])
        input_shape = (self.batch_size, ) + sample_frames.shape

        X_batch = np.empty(input_shape, dtype=np.float32)
        y_batch = np.empty((self.batch_size), dtype=np.int32)
        
        for i, (path, label) in enumerate(zip(batch_paths, batch_labels)):
            # Load the pre-processed numpy array from the file
            frames = np.load(path)
            
            # Normalize pixel values
            frames = frames / 255.0
            
            X_batch[i,] = frames
            y_batch[i] = label

        return X_batch, y_batch

def get_data_paths_and_labels(data_dir):
    """
    Collects all file paths and labels from the processed data directory.
    """
    all_paths = []
    all_labels = []
    
    class_names = sorted(os.listdir(data_dir))
    class_names = [cls for cls in class_names if not cls.startswith('.')]

    label_encoder = LabelEncoder()
    label_encoder.fit(class_names)
    
    # Save the label mapping
    np.save("label_mapping.npy", label_encoder.classes_)
    
    for class_name in class_names:
        class_path = os.path.join(data_dir, class_name)
        if not os.path.isdir(class_path):
            continue
        
        for video_file in os.listdir(class_path):
            if video_file.endswith(".npy"):
                file_path = os.path.join(class_path, video_file)
                all_paths.append(file_path)
                all_labels.append(class_name)
                
    # Encode labels to integers
    y_encoded = label_encoder.transform(all_labels)

    return all_paths, y_encoded