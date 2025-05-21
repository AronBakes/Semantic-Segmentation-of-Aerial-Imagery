

import os
import numpy as np
import cv2
from tqdm import tqdm
import csv
from scipy.stats import entropy
import matplotlib.pyplot as plt
from collections import defaultdict

# === CONFIG ===
IN_DIR = 'dataset-medium'  # expects subfolders images/, elevations/, labels/
OUT_DIR = 'data/chipped'
TILE_SIZE = 300
TRAIN_STRIDE = 100
EVAL_STRIDE = 300  # for val/test
IGNORE_COLOR = (255, 0, 255)
IGNORE_THRESHOLD = 0.0
BACKGROUND_CLASS = 4
BACKGROUND_SKIP_THRESHOLD = 0.95

COLOR_TO_CLASS = {
    (230, 25, 75): 0,     # Building
    (145, 30, 180): 1,    # Clutter
    (60, 180, 75): 2,     # Vegetation
    (245, 130, 48): 3,    # Water
    (255, 255, 255): 4,   # Background
    (0, 130, 200): 5      # Car
}

RARE_CLASSES = [0, 1, 3, 5]  # prioritise: building, clutter, water, car
CLASS_NAMES = ['Building', 'Clutter', 'Vegetation', 'Water', 'Background', 'Car']
CLASS_TO_COLOR = {v: k for k, v in COLOR_TO_CLASS.items()}
NUM_CLASSES = len(CLASS_NAMES)


VAL_FILES = [
    '1726eb08ef_60693DB04DINSPIRE', '1d4fbe33f3_F1BE1D4184INSPIRE', 
    '2ef3a4994a_0CCD105428INSPIRE', 'a1af86939f_F1BE1D4184OPENPIPELINE', 
    'c6d131e346_536DE05ED2OPENPIPELINE'
    ]

TEST_FILES = [
    '12fa5e614f_53197F206FOPENPIPELINE', '520947aa07_8FCB044F58OPENPIPELINE', 
    '57426ebe1e_84B52814D2OPENPIPELINE', '9170479165_625EDFBAB6OPENPIPELINE', 
    'd9161f7e18_C05BA1BC72OPENPIPELINE'
    ]

PROBLEM_REGIONS = {
    "25f1c24f30_EB81FE6E2BOPENPIPELINE": [
        {"x": 4050, "y": 0, "w": 1300, "h": 1050},
        {"x": 3700, "y": 3650, "w": 200, "h": 170},
        {"x": 3525, "y": 3810, "w": 250, "h": 190},
        {"x": 3780, "y": 3580, "w": 200, "h": 160}
    ],
    "39e77bedd0_729FB913CDOPENPIPELINE": [
        {"x": 2900, "y": 2700, "w": 250, "h": 100}
    ],
    "a1af86939f_F1BE1D4184OPENPIPELINE": [
        {"x": 0, "y": 800, "w": 300, "h": 110}
    ]
}

def chip_image(rgb_path, elev_path, label_path, out_prefix, metadata_rows, split):
    rgb = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
    if rgb is None:
        print(f"⚠️ Failed to load RGB image: {rgb_path}")
        return
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

    elev = read_elevation_float32(elev_path)
    if elev is None:
        print(f"⚠️ Failed to load elevation image: {elev_path}")
        return

    label = cv2.imread(label_path, cv2.IMREAD_COLOR)
    if label is None:
        print(f"⚠️ Failed to load label image: {label_path}")
        return

    h, w, _ = rgb.shape
    base_name = os.path.splitext(os.path.basename(rgb_path))[0].replace('-ortho', '')

    stride = TRAIN_STRIDE if split == 'train' else EVAL_STRIDE

    for y in range(0, h - TILE_SIZE + 1, stride):
        for x in range(0, w - TILE_SIZE + 1, stride):
            if overlaps_problem_region(x, y, base_name):
                continue

            rgb_tile = rgb[y:y+TILE_SIZE, x:x+TILE_SIZE, :]
            elev_tile = elev[y:y+TILE_SIZE, x:x+TILE_SIZE]
            label_tile = label[y:y+TILE_SIZE, x:x+TILE_SIZE, :]

            ignore_mask = np.all(label_tile == IGNORE_COLOR, axis=-1)
            ignore_ratio = np.sum(ignore_mask) / (TILE_SIZE * TILE_SIZE)
            if ignore_ratio > IGNORE_THRESHOLD:
                continue

            label_tile_rgb = cv2.cvtColor(label_tile, cv2.COLOR_BGR2RGB)
            label_ids = np.full((TILE_SIZE, TILE_SIZE), -1, dtype=np.int32)
            for color_rgb, class_idx in COLOR_TO_CLASS.items():
                mask = np.all(label_tile_rgb == color_rgb, axis=-1)
                label_ids[mask] = class_idx

            if np.any(label_ids == -1):
                continue

            total_pixels = TILE_SIZE * TILE_SIZE
            counts_array = np.array([(label_ids == i).sum() for i in range(NUM_CLASSES)], dtype=np.float32)
            if counts_array.sum() == 0:
                continue

            class_percentages = counts_array / counts_array.sum()
            building_pct, clutter_pct, vegetation_pct, water_pct, background_pct, car_pct = class_percentages
            class_entropy = entropy(class_percentages + 1e-9, base=2)

            if background_pct == 1.0:
                continue

            if background_pct > BACKGROUND_SKIP_THRESHOLD and not any(counts_array[i] > 0 for i in RARE_CLASSES):
                continue

            tile_id = f'{base_name}_{x}_{y}'

            rgb_out = os.path.join(OUT_DIR, split, 'images', f'{tile_id}-ortho.png')
            elev_out = os.path.join(OUT_DIR, split, 'elevations', f'{tile_id}-elev.npy')
            label_out = os.path.join(OUT_DIR, split, 'labels', f'{tile_id}-label.png')

            cv2.imwrite(rgb_out, cv2.cvtColor(rgb_tile, cv2.COLOR_RGB2BGR))
            np.save(elev_out, elev_tile)
            cv2.imwrite(label_out, label_tile)

            metadata_rows.append([tile_id, base_name, x, y] + class_percentages.tolist() + [float(class_entropy)])

def read_elevation_float32(path):
    tif = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if tif is None:
        return None
    if len(tif.shape) == 3 and tif.shape[2] > 1:
        tif = tif[:, :, 0]
    if tif.dtype != np.float32:
        tif = tif.astype(np.float32)
    return tif

def overlaps_problem_region(x, y, base_name):
    if base_name not in PROBLEM_REGIONS:
        return False
    for region in PROBLEM_REGIONS[base_name]:
        rx, ry, rw, rh = region['x'], region['y'], region['w'], region['h']
        if (x + TILE_SIZE > rx and x < rx + rw and y + TILE_SIZE > ry and y < ry + rh):
            return True
    return False

def chip_all():
    from statistics import mean
    metadata = {'train': [], 'val': [], 'test': []}
    img_dir = os.path.join(IN_DIR, 'images')
    elev_dir = os.path.join(IN_DIR, 'elevations')
    label_dir = os.path.join(IN_DIR, 'labels')

    filenames = [f for f in os.listdir(img_dir) if f.endswith('-ortho.tif')]
    splits = {'train': [], 'val': [], 'test': []}

    for fname in filenames:
        prefix = os.path.splitext(fname)[0].replace('-ortho', '')
        if prefix in TEST_FILES:
            splits['test'].append(prefix)
        elif prefix in VAL_FILES:
            splits['val'].append(prefix)
        else:
            splits['train'].append(prefix)

    for split in ['train', 'val', 'test']:
        print(f"\n🔄 Processing {split} set")
        for prefix in tqdm(splits[split], desc=f"Chipping {split}"):
            rgb_path = os.path.join(img_dir, f'{prefix}-ortho.tif')
            elev_path = os.path.join(elev_dir, f'{prefix}-elev.tif')
            label_path = os.path.join(label_dir, f'{prefix}-label.png')

            if os.path.exists(rgb_path) and os.path.exists(elev_path) and os.path.exists(label_path):
                chip_image(rgb_path, elev_path, label_path, prefix, metadata[split], split)

    for split in ['train', 'val', 'test']:
        csv_path = os.path.join(OUT_DIR, f'{split}_metadata.csv')
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            header = ['tile_id', 'source_file', 'x', 'y'] + [f'{i}: {name}' for i, name in enumerate(CLASS_NAMES)] + ['entropy']
            writer.writerow(header)
            writer.writerows(metadata[split])
        print(f"✅ Metadata saved to {csv_path}")
        print(f"📊 {split.upper()} STATS")
        print(f"  Total tiles: {len(metadata[split])}")
        if metadata[split]:
            entropies = [row[-1] for row in metadata[split]]
            print(f"  Mean entropy: {mean(entropies):.4f}")
            print(f"  Min entropy: {min(entropies):.4f}")
            print(f"  Max entropy: {max(entropies):.4f}")

if __name__ == '__main__':
    if os.path.exists(OUT_DIR):
        response = input(f"⚠️ Output directory '{OUT_DIR}' already exists. Overwrite? [Y/n]: ").strip().lower()
        if response not in ['', 'y', 'yes']:
            print("❌ Aborting chipping process.")
            exit()

    for split in ['train', 'val', 'test']:
        for folder in ['images', 'elevations', 'labels']:
            os.makedirs(os.path.join(OUT_DIR, split, folder), exist_ok=True)

    chip_all()
    print("\n✅ Done! Chipped tiles stored in data/chipped/{train,val,test}/")
