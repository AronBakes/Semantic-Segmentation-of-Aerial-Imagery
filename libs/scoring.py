# libs/scoring.py

import os
import numpy as np
import cv2
from sklearn.metrics import confusion_matrix

def score_predictions(dataset_name, basedir, num_classes=7):
    print("Scoring predictions...")

    pred_dir = os.path.join(basedir, 'predictions')
    gt_dir = 'data/test/labels'

    pred_files = os.listdir(pred_dir)

    total_iou = np.zeros(num_classes)
    total_pixels = np.zeros(num_classes)

    for file_name in pred_files:
        pred = cv2.imread(os.path.join(pred_dir, file_name), cv2.IMREAD_GRAYSCALE)
        gt = cv2.imread(os.path.join(gt_dir, file_name), cv2.IMREAD_GRAYSCALE)

        for cls in range(num_classes):
            pred_mask = (pred == cls)
            gt_mask = (gt == cls)

            intersection = np.logical_and(pred_mask, gt_mask).sum()
            union = np.logical_or(pred_mask, gt_mask).sum()

            if union != 0:
                total_iou[cls] += intersection / union
                total_pixels[cls] += 1

    miou_per_class = total_iou / np.maximum(total_pixels, 1)
    miou = np.mean(miou_per_class)

    print(f"Mean IoU: {miou:.4f}")

    return {'miou': miou}, miou_per_class
