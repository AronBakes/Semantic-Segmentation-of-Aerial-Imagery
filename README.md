# 🛰️ CAB420 - Semantic Segmentation of Aerial Imagery

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=flat-square)
![License](https://img.shields.io/badge/License-Academic-lightgrey?style=flat-square)
![Status](https://img.shields.io/badge/Status-In%20Development-yellow?style=flat-square)

This project investigates how semantic segmentation of aerial imagery can be improved by combining **RGB** and **elevation (DEM)** data. The objective is to classify each pixel into meaningful categories using DL models **U-Net** and **SegFormer**, and non-DL model **CRF**.

---

## 👥 Group Members
- **Aron Bakes** (n11405384)
- **Deegan Marks** (n11548444)
- **Jordan Geltch-Robb** (n11427515)

---

## 📁 Project Structure

```
├── data/                 # Chipped dataset (train/val/test splits)
│   ├── train/
│   ├── val/
│   ├── test/
│   └── tiles_metadata.csv
├── data.ipynb            # Data loading and generator logic
├── models.ipynb          # U-Net and Multi-U-Net model definitions
├── training.ipynb        # Training loop and evaluation
├── scoring.ipynb         # Test set evaluation metrics
├── util.ipynb            # Utility functions and plotting
├── chip_dataset.py       # Optional: preprocessing script to create chips
└── README.md             # Project description
```

---

## 🚀 Getting Started

### 1. Environment Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

> 📦 Dependencies include: `tensorflow`, `numpy`, `opencv-python`, `matplotlib`, etc.

### 2. Prepare Dataset

- Place your raw dataset in `dataset-medium/`
- Run the chipping script to split it into tiles:

```bash
python chip_dataset.py
```

This will populate the folders:
- `data/chipped/train/`
- `data/chipped/val/`
- `data/chipped/test/`

---

## 🧠 Model Training

Open `training.ipynb` and run:

```python
train_model(
    input_type="rgb_elevation",
    model_type="unet",
    batch_size=8,
    epochs=50,
    steps_per_epoch=100
)
```

---

## ✅ Evaluation

### Automatically After Training
Test evaluation runs automatically at the end of training using:

```python
evaluate_on_test(model, test_gen, n_vis=10)
```

### Manual Test Evaluation
You can re-run it any time using:

```python
test_gen = StreamingDataGenerator(...)
evaluate_on_test(model, test_gen, n_vis=10)
```

---

## 🖼️ Sample Predictions

Example output from `visualise_prediction()`:

| RGB Image | Ground Truth | Model Prediction |
|-----------|--------------|------------------|
| ![](docs/sample_rgb.png) | ![](docs/sample_true.png) | ![](docs/sample_pred.png) |

*(Add your own images in a `/docs` folder or embed inline in notebook)*

---

## 🏷️ Class Labels

| ID | Class        | Colour (RGB)     |
|----|--------------|------------------|
| 0  | Building     | (230, 25, 75)     |
| 1  | Clutter      | (145, 30, 180)    |
| 2  | Vegetation   | (60, 180, 75)     |
| 3  | Water        | (245, 130, 48)    |
| 4  | Background   | (255, 255, 255)   |
| 5  | Car          | (0, 130, 200)     |

---

## 📊 Class Distribution Tracking

- `DistributionLogger` tracks class balance per epoch
- Cumulative counts are plotted at the end of training
- Final per-class distribution is printed in percentages

---

## 📌 Notes

- Problem regions are skipped during chipping
- 100% background tiles are discarded
- >95% background tiles are skipped **unless** they contain rare classes
- Evaluation is run on **all test batches**

---

## 📄 License

This repository is part of the CAB420 course at QUT and intended for academic use only.
