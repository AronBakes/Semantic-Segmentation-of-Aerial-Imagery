# Showcase: Semantic Segmentation of Aerial Imagery with RGB and Elevation Data

## Overview
This project builds on the CAB420 Machine Learning Report (June 17, 2025) by enhancing aerial imagery segmentation with RGB and elevation data. It showcases a pre-trained U-Net model on an updated DroneDeploy dataset, shifting from 256x256 to 512x512 images and adding Roads, achieving a 10% mIoU improvement after improving labels. All results are pre-computed, meaning they were generated previously with the 600MB U-net model trained from scratch (not included here), to enable static viewing without memory-intensive execution requiring a GPU.

## Key Updates from the Report
- **Dataset Evolution**: Original dataset (Water, Building, Clutter, Vegetation, Background, Car) updated by adding Roads, removing Clutter due to ambiguity, and swapping Water (now blue, previously orange) and Car colors for better visual clarity, with a 10% mIoU gain.
- **Resolution Boost**: Increased from 256x256 to 512x512 images for better generalization, reducing batch size to a quarter (e.g., from 32 to 8) due to higher memory needs.
- **Model Approach**: U-Net and SegFormer built from scratch without pre-training, as elevation data required custom architectures unsuitable for ImageNet backbones.
- **Class Balance**: Updated labels reduced confusion across Background and Building, though aerial imagery remains inherently imbalanced.

## Data Insights
### Chipping and Original Dataset
The DroneDeploy dataset was chipped into 256x256 tiles, later updated to 512x512. Figure 1 shows a sample: RGB, elevation, and label map.

**Pre-computed Output**:  
- **Sample Image**: ![Sample Dataset](output/sample_dataset.jpg)  

### Class Distribution (Original)
The original dataset’s class distribution highlights severe imbalance, dominated by Background.

**Pre-computed Output**:  
- **Class Distribution**: ![Original Class Distribution](output/original_class_dist.png)  

### Augmentation: CutMix on Water
CutMix was applied to Water to address class imbalance, mixing patches to enhance training.

**Pre-computed Output**:  
- **CutMix on Water**: ![CutMix Water Example](output/cutmix_water.png)  

### Re-labeling the Dataset
The dataset was updated by adding the Road class, removing the ambiguous Clutter class, and swapping Water (now blue, previously orange) and Car colors for clarity. Errors were fixed, and resolution adjusted to 512x512, improving mIoU by 10%. A new label example is shown below. The updated labeled dataset is available on Kaggle [here](https://www.kaggle.com/aronbakes/your-dataset-name).

**Pre-computed Output**:  
- **New Road Label**: ![Road Label Example](output/107f24d6e9_F1BE1D4184INSPIRE-label.png)  

### Class Distribution (Re-labeled)
The re-labeled dataset’s distribution reflects the new class balance after adding Roads, removing Clutter, and adjusting colors.

**Pre-computed Output**:  
- **Re-labeled Class Distribution**: ![Re-labeled Class Distribution](output/relabeled_class_dist.png)  

## Segmentation Models
### Loss Functions
A custom loss combined Categorical Cross-Entropy (0.25), Dice Loss (1.75), and Focal Loss (2.5, gamma=4) to handle imbalance. Class weights, fine-tuned and normalized, were used as alpha in Focal Loss and integrated into Dice Loss to prioritize hard cases. Additionally, 10% label smoothing was applied to improve generalization.

## Evaluation
### RGB vs RGB + Elevation
Elevation data boosted mIoU: U-Net from 0.3985 (RGB) to 0.4295 (RGB+Elevation).

### Confusion Matrices
Confusion matrices compare U-Net performance across configurations. The matrices are displayed side by side for comparison.

| RGB Only (Old Labels)       | RGB+Elevation (Old Labels) | RGB Only (New Labels) |
|-----------------------------|----------------------------|-----------------------|
| ![Confusion Matrix Old RGB](output/confusion_matrix_rgb.png) | ![Confusion Matrix Old Elev](output/confusion_matrix_rgb_elev.png) | ![Confusion Matrix New RGB](output/confusion_matrix_new_labels.png) |

### Reconstruction Results
The U-Net model’s segmentation output is validated through full-tile reconstruction, stitching chipped predictions back into the original image. Figure 2 illustrates a reconstructed output, showcasing the model’s ability to segment classes like Roads and Buildings with improved spatial coherence, particularly with the re-labeled dataset.

**Pre-computed Output**:  
- **Reconstruction Output**: ![Reconstruction Result](output/reconstruction.png)  

### IoU Performance Per Class
Table 1 shows mean IoU and IoU improvements per class before anb after re-labeling.

| Class         | U-Net (RGB) | U-Net (RGB+Elev) | U-Net (RGB, New Labels) |
|---------------|-------------|------------------|-------------------------|
| Building      | 0.3658      | 0.5409           | 0.6184                  |
| Clutter       | 0.1320      | 0.1609           | --                      |
| Vegetation    | 0.5942      | 0.6012           | 0.7417                  |
| Water         | 0.0521      | 0.0548           | 0.4931                  |
| Background    | 0.7046      | 0.7263           | 0.7346                  |
| Car           | 0.5426      | 0.4927           | 0.0620                  |
| Road          | --          | --               | 0.5039                  |
| **mIoU**      | 0.3985      | 0.4295           | 0.5256                  |

## Conclusion
This showcase highlights the U-Net model’s success with 512x512 images and re-labeled data, achieving for a 10% mIoU boost via Roads and Clutter removal. The addition of the Road class allowed the model to better distinguish Background from Building, while removing the Clutter class eliminated class-wide confusion due to Clutter having a vague definition. Elevation enhanced man-made structure detection, with re-labeling and custom loss strategies (class weights, 10% smoothing) refining accuracy. All results are pre-computed using a 600MB model trained from scratch, not included here.

**Note**: Figures are in the `output` folder. The 600MB `segmentation_model.keras` is unavailable; contact the author for access if needed. 
