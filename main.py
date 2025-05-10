# libs/scoring.py

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, jaccard_score

# Class label mapping (index: name)
CLASS_NAMES = [
    "Building",    # 0
    "Clutter",     # 1
    "Vegetation",  # 2
    "Water",       # 3
    "Ground",      # 4
    "Car"           # 5
]

IGNORE_CLASS_INDEX = 6

# Colour-to-class mapping (for reference or decoding colour masks)
COLOR_TO_CLASS = {
    (75, 25, 230): 0,       # BUILDING
    (180, 30, 145): 1,      # CLUTTER
    (75, 180, 60): 2,       # VEGETATION
    (48, 130, 245): 3,      # WATER
    (255, 255, 255): 4,     # GROUND
    (200, 130, 0): 5,       # CAR
    (255, 0, 255): 6        # IGNORE
}


def evaluate_predictions(pred_mask, true_mask):
    """
    Compare predicted and true masks using classification metrics.
    Excludes IGNORE pixels (label = 6).

    Args:
        pred_mask (np.ndarray): predicted labels, shape (H, W)
        true_mask (np.ndarray): ground truth labels, shape (H, W)
    """
    assert pred_mask.shape == true_mask.shape, "Prediction and label shape mismatch"

    # Mask out IGNORE pixels
    valid_mask = (true_mask != IGNORE_CLASS_INDEX)
    y_pred = pred_mask[valid_mask].flatten()
    y_true = true_mask[valid_mask].flatten()

    print("\n--- Classification Report ---")
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES, digits=3))

    print("\n--- Confusion Matrix ---")
    print(confusion_matrix(y_true, y_pred))

    print("\n--- Mean IoU ---")
    miou = jaccard_score(y_true, y_pred, average='macro')
    print(f"Mean IoU (macro): {miou:.3f}")
    return miou
