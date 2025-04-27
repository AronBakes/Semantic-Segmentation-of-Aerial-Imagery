# datasets.py

from sklearn.model_selection import train_test_split
import os
import shutil

def prepare_dataset_structure(base_dir='data'):
    images_dir = os.path.join(base_dir, 'images')
    elevations_dir = os.path.join(base_dir, 'elevations')
    labels_dir = os.path.join(base_dir, 'labels')

    # Check if split already exists
    if os.path.exists(os.path.join(base_dir, 'train')):
        print("Train/Val/Test folders already exist. Skipping split.")
        return

    # List all images
    filenames = os.listdir(images_dir)

    # Shuffle and split
    train_filenames, temp_filenames = train_test_split(filenames, test_size=0.2, random_state=42)
    val_filenames, test_filenames = train_test_split(temp_filenames, test_size=0.5, random_state=42)

    # Create folders
    for split in ['train', 'val', 'test']:
        for subfolder in ['images', 'elevations', 'labels']:
            os.makedirs(os.path.join(base_dir, split, subfolder), exist_ok=True)

    # Move files
    def move_files(file_list, split_name):
        for fname in file_list:
            shutil.copy(os.path.join(images_dir, fname), os.path.join(base_dir, split_name, 'images', fname))
            shutil.copy(os.path.join(elevations_dir, fname), os.path.join(base_dir, split_name, 'elevations', fname))
            shutil.copy(os.path.join(labels_dir, fname), os.path.join(base_dir, split_name, 'labels', fname))

    move_files(train_filenames, 'train')
    move_files(val_filenames, 'val')
    move_files(test_filenames, 'test')

    print("Dataset split into Train / Validation / Test sets successfully.")
