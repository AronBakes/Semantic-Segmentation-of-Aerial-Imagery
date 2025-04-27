# libs/data_generator.py

import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.utils import Sequence
import cv2
import random

class DataGenerator(Sequence):
    def __init__(self, image_dir, elevation_dir, label_dir, file_list, batch_size=32, input_type='rgb', target_size=(256, 256), shuffle=True, num_classes=7):
        self.image_dir = image_dir
        self.elevation_dir = elevation_dir
        self.label_dir = label_dir
        self.file_list = file_list
        self.batch_size = batch_size
        self.input_type = input_type
        self.target_size = target_size
        self.shuffle = shuffle
        self.num_classes = num_classes
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.file_list) / self.batch_size))

    def __getitem__(self, index):
        # Generate indexes for the batch
        batch_files = self.file_list[index * self.batch_size : (index + 1) * self.batch_size]

        # Generate data
        X, y = self.__data_generation(batch_files)

        return X, y

    def on_epoch_end(self):
        if self.shuffle:
            random.shuffle(self.file_list)

    def __data_generation(self, batch_files):
        X = []
        y = []

        for file_name in batch_files:
            # --- Load image ---
            image = cv2.imread(os.path.join(self.image_dir, file_name))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, self.target_size)

            # --- Load elevation ---
            elevation = cv2.imread(os.path.join(self.elevation_dir, file_name), cv2.IMREAD_GRAYSCALE)
            elevation = cv2.resize(elevation, self.target_size)

            # --- Load label ---
            label = cv2.imread(os.path.join(self.label_dir, file_name), cv2.IMREAD_GRAYSCALE)
            label = cv2.resize(label, self.target_size)
            label = tf.keras.utils.to_categorical(label, num_classes=self.num_classes)

            # --- Merge inputs based on input_type ---
            if self.input_type == '1ch':
                merged = np.expand_dims(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY), axis=-1)  # Grayscale only
            elif self.input_type == '2ch':
                grayscale = np.expand_dims(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY), axis=-1)
                elevation = np.expand_dims(elevation, axis=-1)
                merged = np.concatenate([grayscale, elevation], axis=-1)
            elif self.input_type == 'rgb':
                merged = image
            elif self.input_type == 'rgb_elevation':
                elevation = np.expand_dims(elevation, axis=-1)
                merged = np.concatenate([image, elevation], axis=-1)
            else:
                raise ValueError("Invalid input_type.")

            # Normalise image inputs
            merged = merged / 255.0

            X.append(merged)
            y.append(label)

        X = np.array(X)
        y = np.array(y)

        return X, y
