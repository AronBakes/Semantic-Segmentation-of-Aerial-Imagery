# libs/training.py

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TerminateOnNaN
from libs.models import build_unet
from libs.data_generator import DataGenerator
from libs import scoring

# Supported input types
INPUT_TYPE_CONFIG = {
    "1ch": {"description": "grayscale only", "channels": 1},
    "2ch": {"description": "grayscale + elevation", "channels": 2},
    "rgb": {"description": "RGB only", "channels": 3},
    "rgb_elevation": {"description": "RGB + elevation", "channels": 4}
}

def train_model(input_type="rgb_elevation", batch_size=8, epochs=10, tile_size=512):
    assert input_type in INPUT_TYPE_CONFIG, f"Unknown input type: {input_type}"
    num_channels = INPUT_TYPE_CONFIG[input_type]["channels"]

    print(f"\n🔧 Training with input type: {input_type} ({num_channels} channels)")

    # --- Paths ---
    base_dir = "data/chipped"
    train_images = os.path.join(base_dir, "train", "images")
    train_elev = os.path.join(base_dir, "train", "elevations")
    train_labels = os.path.join(base_dir, "train", "labels")

    val_images = os.path.join(base_dir, "val", "images")
    val_elev = os.path.join(base_dir, "val", "elevations")
    val_labels = os.path.join(base_dir, "val", "labels")

    # --- Files ---
    train_files = sorted([f for f in os.listdir(train_images) if f.endswith("-ortho.png")])
    val_files = sorted([f for f in os.listdir(val_images) if f.endswith("-ortho.png")])

    # --- Data Generators ---
    train_gen = DataGenerator(train_images, train_elev, train_labels, train_files,
                              batch_size=batch_size, input_type=input_type, num_classes=6)
    val_gen = DataGenerator(val_images, val_elev, val_labels, val_files,
                            batch_size=batch_size, input_type=input_type, num_classes=6)

    # --- Model ---
    model = build_unet(input_shape=(tile_size, tile_size, num_channels), num_classes=6)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    # --- Callbacks ---
    os.makedirs("checkpoints", exist_ok=True)
    checkpoint = ModelCheckpoint("checkpoints/best_model.h5", monitor='val_accuracy', save_best_only=True)
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)
    nan_terminate = TerminateOnNaN()

    class TimeLimitCallback(tf.keras.callbacks.Callback):
        def __init__(self, max_minutes=20):
            super().__init__()
            self.max_duration = max_minutes * 60
        def on_train_begin(self, logs=None):
            self.start_time = tf.timestamp()
        def on_epoch_end(self, epoch, logs=None):
            elapsed = tf.timestamp() - self.start_time
            if elapsed > self.max_duration:
                print(f"⏱️ Training time exceeded {self.max_duration // 60} minutes. Stopping early.")
                self.model.stop_training = True

    time_limit = TimeLimitCallback(max_minutes=20)

    callbacks = [checkpoint, early_stop, reduce_lr, nan_terminate, time_limit]

    # --- Training ---
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs,
        callbacks=callbacks
    )

    # --- Evaluate ---
    val_imgs, val_lbls = next(iter(val_gen))
    pred = model.predict(val_imgs)
    pred_mask = np.argmax(pred[0], axis=-1)
    true_mask = np.argmax(val_lbls[0], axis=-1)

    print("\n📊 Evaluation Results:")
    scoring.evaluate_predictions(pred_mask, true_mask)
