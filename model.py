import tensorflow as tf
from tensorflow.keras.layers import Dense, LSTM, TimeDistributed, Dropout, Input
from tensorflow.keras.models import Model
from tensorflow.keras.applications import MobileNetV2

# --- Configuration (These should match your preprocessing script) ---
IMG_SIZE = 128
NUM_FRAMES = 30
NUM_CLASSES = 50

def build_model():
    """
    Builds and compiles the hybrid CNN-LSTM model for activity recognition.
    """
    
    # 1. Define the single input layer for our model
    # The input will be a sequence of images: (NUM_FRAMES, IMG_SIZE, IMG_SIZE, 3)
    model_input = Input(shape=(NUM_FRAMES, IMG_SIZE, IMG_SIZE, 3))
    
    # 2. Create the CNN base model (Feature Extractor)
    # We will use MobileNetV2 pre-trained on ImageNet without the top classification layer.
    cnn_base = MobileNetV2(
        weights="imagenet",
        include_top=False,
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )
    
    # Freeze the layers of the base CNN model so they don't get trained.
    cnn_base.trainable = False
    
    # 3. Create the TimeDistributed wrapper
    # This applies the same CNN to each frame in our sequence.
    x = TimeDistributed(cnn_base)(model_input)
    
    # 4. Global Average Pooling to flatten the CNN output for each frame
    # This converts the 3D output of the CNN into a 1D vector for each frame.
    x = TimeDistributed(tf.keras.layers.GlobalAveragePooling2D())(x)
    
    # 5. Create the LSTM network (Sequence Modeler)
    # The LSTM layer will process the sequence of feature vectors.
    x = LSTM(64, return_sequences=False)(x)
    
    # 6. Add a Dropout layer for regularization to prevent overfitting
    x = Dropout(0.5)(x)
    
    # 7. Add the final classification layer
    # This Dense layer has NUM_CLASSES units and a softmax activation function
    # to output the probabilities for each activity class.
    output_layer = Dense(NUM_CLASSES, activation='softmax')(x)
    
    # 8. Create the final model
    # Now, the inputs and outputs are correctly connected through the 'x' variable.
    model = Model(inputs=model_input, outputs=output_layer)
    
    # 9. Compile the model
    # We use the Adam optimizer and SparseCategoricalCrossentropy loss,
    # which is ideal for multi-class classification with integer labels.
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

if __name__ == "__main__":
    # Create the model instance and print a summary to verify the architecture
    model = build_model()
    model.summary()