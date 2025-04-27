# main.py

from libs import training_keras
from libs import datasets
from libs import models_keras
from libs import inference_keras
from libs import scoring

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
        model = models_keras.build_unet(input_shape=(256, 256, input_channels), num_classes=7)
    elif args.model == 'segformer':
        model = models_keras.build_segformer(input_shape=(256, 256, input_channels), num_classes=7)
    else:
        raise ValueError("Invalid model choice.")

    # --- Train Model ---
    training_keras.train_model(args.dataset, model, input_type=args.input_type)

    # --- Run Inference ---
    inference_keras.run_inference(args.dataset, model=model, basedir=wandb.run.dir)

    # --- Score Predictions ---
    score, _ = scoring.score_predictions(args.dataset, basedir=wandb.run.dir)
    print(score)
    wandb.log(score)
