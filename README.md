# CAB420 Semantic Segmentation Project

This repository contains our CAB420 Group Project for Semantic Segmentation of Aerial Drone Data.

---

## 📦 Project Structure

| Folder/File | Description |
|:------------|:------------|
| `/libs` | Python code (data splitters, model builders, etc.) |
| `/output` | (Optional) Output folder for logs/results |
| `main.py` | Main training script |
| `jupyter.ipynb` | Example or experimentation notebook |
| `README.md` | Project overview |
| `.gitignore` | Git ignore rules |

---

## 📥 Downloading the Dataset

Download the DroneDeploy Medium Dataset from Kaggle:

**[DroneDeploy Medium Dataset (Kaggle)](https://www.kaggle.com/datasets/mightyrains/drone-deploy-medium-dataset?resource=download-directory)**

After downloading:
- Extract the dataset to the project root
- Rename the folder to `dataset-medium`
- Inside `dataset-medium/` you should have:
  - `images/`
  - `elevations/`
  - `labels/`
  - `index.csv`

---

## 🛠️ Preparing the Dataset

To split the dataset into `train`, `val`, and `test`:

```bash
python3 libs/datasets.py
# CAB420 - Semantic Segmentation of Aerial Imagery

This project investigates how semantic segmentation of aerial imagery can be improved by combining RGB and elevation data.

## Group Members
- Aron Bakes (n11405384)
- Deegan Marks (n11548444)
- Jordan Geltch-Robb (n11427515)

## Project Structure
- Python scripts and Jupyter notebooks are located in this folder.
- Datasets (if any) are inside `Data/` subfolder.

## Run Instructions
- Open the `.ipynb` notebooks with Jupyter or VS Code.
- Alternatively, run `.py` scripts directly.

# Hello
