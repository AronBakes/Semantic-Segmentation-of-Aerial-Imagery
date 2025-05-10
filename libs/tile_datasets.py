# libs/tile_datasets.py

import os
import numpy as np
import cv2
from tqdm import tqdm

# === CONFIG ===
IN_DIR = 'data'  # expects subfolders train/, val/
OUT_DIR = 'data/chipped'
TILE_SIZE = 512
STRIDE = 256  # overlap added
IGNORE_COLOR = (255, 0, 255)
IGNORE_THRESHOLD = 0.0  # reject tiles with any IGNORE


def read_elevation_float32(path):
    tif = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if tif is None:
        return None
    # Handle only 1 channel float32 data
    if len(tif.shape) == 3 and tif.shape[2] > 1:
        tif = tif[:, :, 0]
    if tif.dtype != np.float32:
        tif = tif.astype(np.float32)
    return tif


def chip_image(rgb_path, elev_path, label_path, out_prefix, split):
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

    for y in range(0, h - TILE_SIZE + 1, STRIDE):
        for x in range(0, w - TILE_SIZE + 1, STRIDE):
            rgb_tile = rgb[y:y+TILE_SIZE, x:x+TILE_SIZE, :]
            elev_tile = elev[y:y+TILE_SIZE, x:x+TILE_SIZE]
            label_tile = label[y:y+TILE_SIZE, x:x+TILE_SIZE, :]

            ignore_mask = np.all(label_tile == IGNORE_COLOR, axis=-1)
            ignore_ratio = np.sum(ignore_mask) / (TILE_SIZE * TILE_SIZE)
            if ignore_ratio > IGNORE_THRESHOLD:
                continue

            tile_id = f'{out_prefix}_{y}_{x}'
            rgb_out = os.path.join(OUT_DIR, split, 'images', f'{tile_id}-ortho.png')
            elev_out = os.path.join(OUT_DIR, split, 'elevations', f'{tile_id}-elev.npy')
            label_out = os.path.join(OUT_DIR, split, 'labels', f'{tile_id}-label.png')

            cv2.imwrite(rgb_out, cv2.cvtColor(rgb_tile, cv2.COLOR_RGB2BGR))
            np.save(elev_out, elev_tile)
            cv2.imwrite(label_out, label_tile)


def chip_all():
    for split in ['train', 'val']:  # test skipped by design
        print(f'\nChipping {split} set')
        split_dir = os.path.join(IN_DIR, split)
        img_dir = os.path.join(split_dir, 'images')
        elev_dir = os.path.join(split_dir, 'elevations')
        label_dir = os.path.join(split_dir, 'labels')

        filenames = [f for f in os.listdir(img_dir) if f.endswith('-ortho.tif')]
        for fname in tqdm(filenames):
            prefix = os.path.splitext(fname)[0]
            prefix = prefix.replace('-ortho', '')  # remove suffix to match elev/label

            rgb_path = os.path.join(img_dir, f'{prefix}-ortho.tif')
            elev_path = os.path.join(elev_dir, f'{prefix}-elev.tif')
            label_path = os.path.join(label_dir, f'{prefix}-label.png')

            if os.path.exists(rgb_path) and os.path.exists(elev_path) and os.path.exists(label_path):
                chip_image(rgb_path, elev_path, label_path, prefix, split)
            else:
                print(f"⚠️ Skipping {prefix} — missing one of RGB/elevation/label")


if __name__ == '__main__':
    for split in ['train', 'val']:
        for folder in ['images', 'elevations', 'labels']:
            os.makedirs(os.path.join(OUT_DIR, split, folder), exist_ok=True)

    chip_all()
    print("\n✅ Done! Chipped tiles stored in data/chipped/")
