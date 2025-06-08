# 🛰️ CAB420 - Semantic Segmentation of Aerial Imagery

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=flat-square)
![License](https://img.shields.io/badge/License-Academic-lightgrey?style=flat-square)
![Status](https://img.shields.io/badge/Status-Complete-green?style=flat-square)

This project investigates the use of RGB and elevation (DEM) data to improve semantic segmentation of aerial imagery. Using both convolutional and transformer-based architectures, we evaluate the performance impact of multimodal fusion on per-pixel classification accuracy.

---

## 👥 Group Members

- **Aron Bakes** (n11405384)
- **Deegan Marks** (n11548444)
- **Jordan Geltch-Robb** (n11427515)

---

## 📦 Dataset

This project uses the [Aerial Semantic Segmentation Dataset](https://drive.google.com/file/d/1FiQQ-fKHpBsOq0sp2e-GxNUtQSgvzAOY/view?usp=sharing), provided for CAB420 coursework. The dataset includes RGB, elevation, and slope maps with pixel-level annotations across six classes: building, clutter, vegetation, water, background, and car.

> **Note:** Dataset contains annotation inconsistencies (e.g. partial/mixed labelling of cars as clutter).

---

## 📁 Project Structure

```
├── _main.ipynb           # Project entry point and main training loop
├── callbacks.ipynb       # Custom callbacks (early stopping, metrics)
├── data.ipynb            # Dataset generation, augmentations, loading
├── distribute.ipynb      # Strategy for training across multiple devices
├── inference.ipynb       # Visualisation and prediction for test data
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
| 1  | Clutter    | (145, 30, 180)    |
| 2  | Vegetation | (60, 180, 75)     |
| 3  | Water      | (245, 130, 48)    |
| 4  | Background | (255, 255, 255)   |
| 5  | Car        | (0, 130, 200)     |

---

## 📌 Notes

- Elevation/slope improves segmentation of buildings, roads, and water.
- Loss functions use CCE + Dice + Focal with tuned weights.
- Dataset contains inconsistencies (e.g. dual-labelling of cars/clutter).
- All models trained and tested on 256×256 image tiles.

---

## 📄 License

This project is licensed for educational use as part of CAB420 at QUT. See the [LICENSE](LICENSE) file for details.

