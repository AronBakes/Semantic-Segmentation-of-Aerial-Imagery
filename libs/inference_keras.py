# libs/inference_keras.py

import os
import numpy as np
import cv2
from tqdm import tqdm

def run_inference(dataset_name, model, basedir, input_type='rgb', target_size=(256, 256)):
    print("Running inference on test set...")

    test_images_dir = 'data/test/images'
    test_elevations_dir = 'data/test/elevations'
    save_dir = os.path.join(basedir, 'predictions')
    os.makedirs(save_dir, exist_ok=True)

    test_files = os.listdir(test_images_dir)

    for file_name in tqdm(test_files):
        # Load image
        image = cv2.imread(os.path.join(test_images_dir, file_name))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, target_size)

        elevation = cv2.imread(os.path.join(test_elevations_dir, file_name), cv2.IMREAD_GRAYSCALE)
        elevation = cv2.resize(elevation, target_size)

        # Handle input_type
        if input_type == '1ch':
            input_tensor = np.expand_dims(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY), axis=-1)
        elif input_type == '2ch':
            grayscale = np.expand_dims(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY), axis=-1)
            elevation = np.expand_dims(elevation, axis=-1)
            input_tensor = np.concatenate([grayscale, elevation], axis=-1)
        elif input_type == 'rgb':
            input_tensor = image
        elif input_type == 'rgb_elevation':
            elevation = np.expand_dims(elevation, axis=-1)
            input_tensor = np.concatenate([image, elevation], axis=-1)

        input_tensor = input_tensor / 255.0
        input_tensor = np.expand_dims(input_tensor, axis=0)  # Batch dimension

        # Predict
        prediction = model.predict(input_tensor)
        prediction = np.argmax(prediction, axis=-1)[0]

        # Save prediction
        save_path = os.path.join(save_dir, file_name)
        cv2.imwrite(save_path, prediction)

    print(f"Inference complete. Predictions saved to {save_dir}.")
