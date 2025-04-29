# main.py

from sklearn.model_selection import train_test_split
import os
import shutil

from libs import training
from libs import datasets
from libs import models
from libs import inference
from libs import scoring
from libs import util

import wandb
import argparse

if __name__ == '__main__':
    # --- Argument Parser ---
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='unet', choices=['unet', 'segformer'], help='Model to train')
    parser.add_argument('--input_type', type=str, default='rgb', choices=['1ch', '2ch', 'rgb', 'rgb_elevation'], help='Input data type')
    parser.add_argument('--dataset', type=str, default='dataset-sample', help='Dataset name')
    args = parser.parse_args()

    # --- Configuration ---
    config = {
        'name': f'{args.model}-{args.input_type}',
        'dataset': args.dataset,
        'input_type': args.input_type,
    }

    # --- Initialise WandB ---
    wandb.init(project="cab420-semantic-segmentation", config=config)

    # --- Prepare Dataset ---
    datasets.prepare_dataset_structure()  # Correct place to call splitting once

    # --- Build Model ---
    input_channels = {
        '1ch': 1,
        '2ch': 2,
        'rgb': 3,
        'rgb_elevation': 4
    }[args.input_type]

    if args.model == 'unet':
        model = models.build_unet(input_shape=(256, 256, input_channels), num_classes=7)
    elif args.model == 'segformer':
        model = models.build_segformer(input_shape=(256, 256, input_channels), num_classes=7)
    else:
        raise ValueError("Invalid model choice.")

    # --- Train Model ---
    training.train_model(args.dataset, model, input_type=args.input_type)

    # --- Run Inference ---
    inference.run_inference(args.dataset, model=model, basedir=wandb.run.dir)

    # --- Score Predictions ---
    score, _ = scoring.score_predictions(args.dataset, basedir=wandb.run.dir)
    print(score)
    wandb.log(score)


color_to_class = {
    (75, 25, 230): 0,       # BUILDING
    (180, 30, 145): 1,      # CLUTTER
    (75, 180, 60): 2,       # VEGETATION
    (48, 130, 245): 3,      # WATER
    (255, 255, 255): 4,     # GROUND
    (200, 130, 0): 5,       # CAR
    (255, 0, 255): 6        # IGNORE
}



'''
def convert_color_to_class(image):
    """
    Convert a color image to a class label image
    """
    # Create an empty image with the same shape as the input
    class_image = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)

    # Iterate through the color to class mapping
    for color, class_id in color_to_class.items():
        # Create a mask for the current color
        mask = np.all(image == np.array(color), axis=-1)
        # Assign the class ID to the corresponding pixels in the class image
        class_image[mask] = class_id

    return class_image
def convert_class_to_color(image):
    """
    Convert a class label image to a color image
    """
    # Create an empty image with the same shape as the input
    color_image = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)

    # Iterate through the class to color mapping
    for color, class_id in color_to_class.items():
        # Create a mask for the current class ID
        mask = (image == class_id)
        # Assign the color to the corresponding pixels in the color image
        color_image[mask] = np.array(color)

    return color_image
    '''