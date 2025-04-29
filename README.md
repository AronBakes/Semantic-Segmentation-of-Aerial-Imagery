# CAB420 Semantic Segmentation Project

This repository contains our CAB420 Group Project for Semantic Segmentation of Aerial Drone Data.

---

## 📦 Project Structure

| Folder/File | Description |
|:------------|:------------|
| `/libs` | Python code (data splitters, model builders, etc.) |
| `/output` | Output folder for figures/results |
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
```

After running this you should see the output:  

📂 TRAIN: 44 images | 44 elevations | 44 labels

📂 VAL: 5 images | 5 elevations | 5 labels

📂 TEST: 6 images | 6 elevations | 6 labels


## Group Members
- Aron Bakes (n11405384)
- Deegan Marks (n11548444)
- Jordan Geltch-Robb (n11427515)


## Project Structure
- Python scripts and Jupyter notebooks are located in this folder.

## Run Instructions
- Open the `.ipynb` notebooks with Jupyter or VS Code.
- Alternatively, run `.py` scripts directly.

