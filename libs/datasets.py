# libs/datasets.py

import os
import shutil
from sklearn.model_selection import train_test_split
import numpy as np
import cv2
import time
from tqdm import tqdm

total_files_copied = 0 


def prepare_dataset_structure(raw_base_dir='dataset-medium', output_base_dir='data'):
    if not os.path.exists(raw_base_dir):
        print(f"❌ Error: Dataset folder '{raw_base_dir}' not found.")
        print("ℹ️ Please download and extract the dataset from:")
        print("   https://www.kaggle.com/datasets/mightyrains/drone-deploy-medium-dataset?resource=download-directory")
        exit(1)

    images_dir = os.path.join(raw_base_dir, 'images')
    elevations_dir = os.path.join(raw_base_dir, 'elevations')
    labels_dir = os.path.join(raw_base_dir, 'labels')

    if os.path.exists(output_base_dir):
        confirm = input(f"⚠️ Warning: '{output_base_dir}' already exists. Delete and recreate it? (y/n): ")
        if confirm.lower() == 'y':
            shutil.rmtree(output_base_dir)
            print(f"🗑️ Deleted '{output_base_dir}'. Starting fresh...")
        else:
            print("❌ Split cancelled by user.")
            exit(0)

    # List all ortho (image) files
    image_files = [f for f in os.listdir(images_dir) if f.endswith('-ortho.tif')]

    print(f"Found {len(image_files)} total images.")

    # Extract prefixes
    prefixes = [f.replace('-ortho.tif', '') for f in image_files]

    # Split into train/val/test
    train_prefixes, temp_prefixes = train_test_split(prefixes, test_size=0.2, random_state=42)
    val_prefixes, test_prefixes = train_test_split(temp_prefixes, test_size=0.5, random_state=42)

    print(f"Splitting {len(prefixes)} total tiles into:")
    print(f"  - {len(train_prefixes)} train tiles")
    print(f"  - {len(val_prefixes)} val tiles")
    print(f"  - {len(test_prefixes)} test tiles")

    # Move files
    move_files(train_prefixes, 'train', images_dir, elevations_dir, labels_dir, output_base_dir)
    move_files(val_prefixes, 'val', images_dir, elevations_dir, labels_dir, output_base_dir)
    move_files(test_prefixes, 'test', images_dir, elevations_dir, labels_dir, output_base_dir)

    print(f"✅ Dataset split complete! New folders created under '{output_base_dir}'.")

def move_files(prefix_list, split_name, images_dir, elevations_dir, labels_dir, output_base_dir):
    global total_files_copied

    for prefix in tqdm(prefix_list, desc=f"Copying {split_name} files"):
        image_fname = prefix + '-ortho.tif'
        elevation_fname = prefix + '-elev.tif'
        label_fname = prefix + '-label.png'

        image_path = os.path.join(images_dir, image_fname)
        elevation_path = os.path.join(elevations_dir, elevation_fname)
        label_path = os.path.join(labels_dir, label_fname)

        # Create folders
        for subfolder in ['images', 'elevations', 'labels']:
            os.makedirs(os.path.join(output_base_dir, split_name, subfolder), exist_ok=True)

        # Copy image
        if os.path.exists(image_path):
            shutil.copy(image_path, os.path.join(output_base_dir, split_name, 'images', image_fname))
            total_files_copied += 1
        else:
            print(f"⚠️ Warning: Missing image {image_fname}")

        # Copy elevation
        if os.path.exists(elevation_path):
            shutil.copy(elevation_path, os.path.join(output_base_dir, split_name, 'elevations', elevation_fname))
            total_files_copied += 1
        else:
            print(f"⚠️ Warning: Missing elevation {elevation_fname}")

        # Copy label
        if os.path.exists(label_path):
            shutil.copy(label_path, os.path.join(output_base_dir, split_name, 'labels', label_fname))
            total_files_copied += 1
        else:
            print(f"⚠️ Warning: Missing label {label_fname}")


def check_split(base_dir='data'):
    splits = ['train', 'val', 'test']

    for split in splits:
        split_path = os.path.join(base_dir, split)

        images_path = os.path.join(split_path, 'images')
        elevations_path = os.path.join(split_path, 'elevations')
        labels_path = os.path.join(split_path, 'labels')

        num_images = len(os.listdir(images_path)) if os.path.exists(images_path) else 0
        num_elevations = len(os.listdir(elevations_path)) if os.path.exists(elevations_path) else 0
        num_labels = len(os.listdir(labels_path)) if os.path.exists(labels_path) else 0

        print(f"\n📂 {split.upper()}: {num_images} images | {num_elevations} elevations | {num_labels} labels")

        if num_images != num_elevations or num_images != num_labels:
            print(f"⚠️ Warning: Count mismatch in {split}!")



if __name__ == '__main__':
    start_time = time.time()  # ⏱️ Start timer

    prepare_dataset_structure()
    check_split()

    elapsed_time = time.time() - start_time
    print(f"\n✅ Dataset split complete in {elapsed_time:.2f} seconds.")
