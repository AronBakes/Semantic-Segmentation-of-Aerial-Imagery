# 🛰️ Semantic Segmentation of Aerial Imagery - Drone Deploy

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)
![Status](https://img.shields.io/badge/Status-Complete-green?style=flat-square)

This project investigates the use of RGB and elevation (DEM) data to improve semantic segmentation of aerial imagery. Using both convolutional and transformer-based architectures, we evaluate the performance impact of multimodal fusion on per-pixel classification accuracy.

---

## 👥 Contributors

- **Aron Bakes** 
- **Deegan Marks** 
- **Jordan Geltch-Robb** 

---

## 📦 Dataset

This project uses the [Aerial Semantic Segmentation Dataset](https://www.kaggle.com/datasets/mightyrains/drone-deploy-medium-dataset). The dataset includes RGB, elevation, and slope maps with pixel-level annotations across six classes: building, clutter, vegetation, water, background, and car.

> **Note:** Dataset contains annotation inconsistencies (e.g. partial/mixed labelling of cars as clutter).

---

## 📁 Project Structure

```
├── _main.ipynb           # Project entry point and main training loop
├── callbacks.ipynb       # Custom callbacks (early stopping, metrics)
├── data.ipynb            # Dataset generation, augmentations, loading
├── distribute.ipynb      # Strategy for training across multiple devices
├── models.ipynb          # Model definitions for U-Net and SegFormer
├── scoring.ipynb         # Evaluation metrics (IoU, F1, etc.)
├── segformer.ipynb       # SegFormer architecture implementation
├── training.ipynb        # Training configuration and execution
├── util.ipynb            # Utility functions and visualisation tools
├── scene_metadata.csv    # Tile-level metadata (test set)
├── train_metadata.csv    # Tile-level metadata (train/val)
└── README.md             # This file
```

---

## 🚀 Getting Started

### 1. Environment Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Dependencies include `tensorflow`, `numpy`, `opencv-python`, `matplotlib`, and others.

### 2. Prepare Dataset

Place raw imagery and labels in `dataset/`, then chip and structure with:

```bash
python chip_dataset.py
```

Expected output directories:
- `data/chipped/train/`
- `data/chipped/val/`
- `data/chipped/test/`

---

## 🧠 Model Training

Launch from `_main.ipynb` and select model/config:

```python
train_model(
    model_type='segformer',        # or 'unet'
    input_type='rgb_elev',         # or 'rgb'
    epochs=50,
    batch_size=8
)
```

---

## ✅ Evaluation

Automatic evaluation runs at the end of training. To manually re-evaluate:

```python
evaluate_on_test(model, test_gen, n_vis=10)
```

---

## 📷 Sample Predictions

| RGB Image | Ground Truth | Prediction |
|-----------|--------------|------------|
| ![](docs/sample_rgb.png) | ![](docs/sample_gt.png) | ![](docs/sample_pred.png) |

---

## 🏷️ Class Labels

| ID | Class      | Colour (RGB)     |
|----|------------|------------------|
| 0  | Building   | (230, 25, 75)     |
| 2  | Vegetation | (60, 180, 75)     |
| 3  | Water      | (0, 130, 200)     |
| 4  | Background | (255, 255, 255)   |
| 5  | Car        | (245, 130, 48)    |
| 5  | Road       | (128, 128, 128)     |

---

## 📌 Notes

- Elevation/slope improves segmentation of buildings, roads, and water.
- Loss functions use CCE + Dice + Focal with tuned weights.
- Dataset contains inconsistencies (e.g. dual-labelling of cars/clutter).
- All models trained and tested on 512×512 image tiles.

---

## 📄 License

This project's code is released under the MIT License. You are free to use, modify, and distribute the code, provided the original copyright and license notice are included. See the [LICENSE](LICENSE) file for details.

