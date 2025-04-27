# libs/training_keras.py

import os
from libs.data_generator import DataGenerator

def train_model(dataset_name, model, input_type='rgb', batch_size=32, epochs=30, target_size=(256, 256), num_classes=7):
    print(f"Starting training: {dataset_name} with input type {input_type}")

    # Set paths
    base_path = 'data'  # Modify if you change your data folder
    train_images = os.path.join(base_path, 'train/images')
    train_elevations = os.path.join(base_path, 'train/elevations')
    train_labels = os.path.join(base_path, 'train/labels')

    val_images = os.path.join(base_path, 'val/images')
    val_elevations = os.path.join(base_path, 'val/elevations')
    val_labels = os.path.join(base_path, 'val/labels')

    # List files
    train_files = os.listdir(train_images)
    val_files = os.listdir(val_images)

    # Create Data Generators
    train_gen = DataGenerator(train_images, train_elevations, train_labels, train_files,
                              batch_size=batch_size, input_type=input_type, target_size=target_size, num_classes=num_classes)
    val_gen = DataGenerator(val_images, val_elevations, val_labels, val_files,
                            batch_size=batch_size, input_type=input_type, target_size=target_size, num_classes=num_classes)

    # Compile model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Fit model
    model.fit(train_gen,
              validation_data=val_gen,
              epochs=epochs)

    # Save model
    model.save(f"trained_{dataset_name}_{input_type}.h5")

    print("Training complete. Model saved.")
