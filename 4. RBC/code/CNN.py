from tensorflow import keras
def create_rbc_model(input_shape, num_classes):
    model = keras.Sequential([
        # Data augmentation
        # keras.layers.RandomFlip("horizontal"),
        # keras.layers.RandomRotation(0.1),
        # keras.layers.RandomZoom(0.1),
        
        # Feature extraction
        keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same', 
                            input_shape=input_shape),
        keras.layers.BatchNormalization(),
        keras.layers.AveragePooling2D((2, 2)),
        keras.layers.Dropout(0.2),
        
        keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        keras.layers.BatchNormalization(),
        keras.layers.AveragePooling2D((2, 2)),
        keras.layers.Dropout(0.3),
        
        keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        keras.layers.BatchNormalization(),
        keras.layers.AveragePooling2D((2, 2)),
        keras.layers.Dropout(0.4),
        
        # Classifier
        keras.layers.Flatten(),
        keras.layers.Dense(256, activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(num_classes, activation='softmax')
    ])
    return model