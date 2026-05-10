"""
convmixer_keras.py
------------------
ConvMixer architecture implemented in Keras/TensorFlow.
Paper: "Patches Are All You Need?" (Trockman & Kolter, TMLR 2022)
https://arxiv.org/abs/2201.09792

Architecture
------------
ConvMixer is an isotropic architecture that:
1. Splits the image into patches via a strided convolution (patch embedding)
2. Applies D repeated "mixer" blocks, each containing:
   a. Depthwise conv (spatial mixing within each channel independently)
   b. Pointwise conv 1×1 (channel mixing across all channels)
   Both with residual connections and GELU activation
3. Global average pooling → FC → softmax

Key insight: maintains the same resolution and number of channels (h)
throughout all D layers — unlike ResNets which downsample progressively.

Default config (ConvMixer-256/8):
    h    = 256   (hidden dimension = number of channels after patch embed)
    D    = 8     (depth = number of mixer blocks)
    patch_size = 7
    kernel_size = 9 (depthwise conv kernel)

Dataset loading and augmentation use minicv — framework only used for
the model definition and training loop.

Usage
-----
    python models/convmixer_keras.py
    python models/convmixer_keras.py --h 128 --depth 6 --epochs 30
"""

import os
import sys
import csv
import json
import numpy as np
from datetime import datetime

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"   # suppress TF info logs

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ===========================================================================
# ConvMixer block
# ===========================================================================

def convmixer_block(x, kernel_size: int, activation=layers.Activation("gelu")):
    """
    One ConvMixer mixer block.

    Steps
    -----
    1. Depthwise conv (spatial mixing):
           groups = h → each channel convolved independently
           kernel = (kernel_size, kernel_size)
       + residual connection + GELU + BatchNorm

    2. Pointwise conv 1×1 (channel mixing):
           standard conv with kernel (1,1)
       + GELU + BatchNorm

    Parameters
    ----------
    x           : Keras tensor  (N, H, W, h)
    kernel_size : int           depthwise conv kernel size
    activation  : Keras layer   activation function

    Returns
    -------
    Keras tensor  (N, H, W, h)  same shape as input
    """
    h = x.shape[-1]

    # Depthwise conv + residual
    residual = x
    x = layers.DepthwiseConv2D(
        kernel_size=kernel_size,
        padding="same",
        use_bias=True,
    )(x)
    x = layers.Activation("gelu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Add()([x, residual])     # residual connection

    # Pointwise conv (channel mixing)
    x = layers.Conv2D(h, kernel_size=1, use_bias=True)(x)
    x = layers.Activation("gelu")(x)
    x = layers.BatchNormalization()(x)

    return x


# ===========================================================================
# ConvMixer model builder
# ===========================================================================

def build_convmixer(
    input_shape: tuple = (96, 96, 3),
    h:           int   = 256,
    depth:       int   = 8,
    patch_size:  int   = 7,
    kernel_size: int   = 9,
    n_classes:   int   = 6,
) -> keras.Model:
    """
    Build the ConvMixer model.

    Parameters
    ----------
    input_shape : tuple  (H, W, C)  — channels-last (Keras default)
    h           : int    hidden dimension (number of channels after patch embed)
    depth       : int    number of ConvMixer blocks
    patch_size  : int    patch embedding stride and kernel size
    kernel_size : int    depthwise conv kernel size in mixer blocks
    n_classes   : int    number of output classes

    Returns
    -------
    keras.Model

    Architecture summary
    --------------------
    Input (H, W, 3)
    → Conv2D(h, patch_size, stride=patch_size) + GELU + BN   [patch embedding]
    → [DepthwiseConv + Residual + GELU + BN,
       Conv1×1 + GELU + BN] × depth                          [mixer blocks]
    → GlobalAvgPool2D
    → Dense(n_classes) + Softmax
    """
    inputs = keras.Input(shape=input_shape)

    # Patch embedding — non-overlapping patches via strided conv
    x = layers.Conv2D(
        filters=h,
        kernel_size=patch_size,
        strides=patch_size,
        padding="valid",
        use_bias=True,
    )(inputs)
    x = layers.Activation("gelu")(x)
    x = layers.BatchNormalization()(x)

    # D mixer blocks
    for _ in range(depth):
        x = convmixer_block(x, kernel_size=kernel_size)

    # Classification head
    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(n_classes, activation="softmax")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="ConvMixer")
    return model


# ===========================================================================
# Data loading — uses minicv-processed .npz files
# ===========================================================================

def load_split(processed_dir: str, split: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Load a preprocessed split from .npz.

    Parameters
    ----------
    processed_dir : str   path to data/processed/
    split         : str   'train_augmented', 'val', or 'test'

    Returns
    -------
    X : np.ndarray  float32 (N, H, W, 3)  channels-last, values in [0, 1]
    y : np.ndarray  int32   (N,)
    """
    path = os.path.join(processed_dir, f"{split}.npz")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Split not found: '{path}'")

    data = np.load(path, allow_pickle=True)
    X    = data["images"].astype(np.float32)   # already (N, H, W, 3) from preprocessing
    y    = data["labels"].astype(np.int32)
    return X, y


# ===========================================================================
# Keras augmentation layer (GPU-accelerated, applied during training only)
# These are simple TF ops — image loading and initial augmentation
# was already done by minicv in augmentor.py
# ===========================================================================

def build_augmentation_layer() -> keras.Sequential:
    """
    Lightweight additional augmentation applied inside the Keras training loop.
    Runs on GPU during training — does NOT replace minicv augmentation.
    Applied only to the training batch, not val/test.

    Transforms: random horizontal flip + small random rotation.
    """
    return keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.05),    # ±5% of 2π ≈ ±18°
    ], name="augmentation")


# ===========================================================================
# Training
# ===========================================================================

def train_convmixer(
    processed_dir: str  = "data/processed",
    log_dir:       str  = "logs",
    h:             int  = 256,
    depth:         int  = 8,
    patch_size:    int  = 7,
    kernel_size:   int  = 9,
    n_classes:     int  = 6,
    lr:            float = 1e-3,
    epochs:        int  = 50,
    batch_size:    int  = 64,
    patience:      int  = 10,
    run_name:      str  = None,
    verbose:       bool = True,
) -> dict:
    """
    Full ConvMixer training pipeline.

    Loads minicv-preprocessed data, builds model, trains with:
    - Adam optimizer
    - Cosine decay LR schedule
    - Early stopping on val_loss
    - CSV logging compatible with other models

    Parameters
    ----------
    processed_dir : str    path to data/processed/ containing .npz files
    log_dir       : str    where to save logs and checkpoints
    h             : int    ConvMixer hidden dim
    depth         : int    number of mixer blocks
    patch_size    : int    patch embedding size
    kernel_size   : int    depthwise conv kernel
    n_classes     : int    output classes
    lr            : float  initial learning rate
    epochs        : int    maximum training epochs
    batch_size    : int    images per mini-batch
    patience      : int    early stopping patience
    run_name      : str    log file prefix
    verbose       : bool   print epoch summaries

    Returns
    -------
    dict  history: epoch, train_loss, val_loss, train_acc, val_acc, lr
    """
    if run_name is None:
        run_name = f"convmixer_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    os.makedirs(log_dir, exist_ok=True)
    log_path  = os.path.join(log_dir, f"{run_name}_logs.csv")
    ckpt_path = os.path.join(log_dir, f"{run_name}_best.keras")
    cfg_path  = os.path.join(log_dir, f"{run_name}_config.json")

    # ---- Load data ----
    print("Loading data...")
    X_train, y_train = load_split(processed_dir, "train_augmented")
    X_val,   y_val   = load_split(processed_dir, "val")
    X_test,  y_test  = load_split(processed_dir, "test")

    print(f"  Train: {X_train.shape}  Val: {X_val.shape}  Test: {X_test.shape}")

    input_shape = X_train.shape[1:]   # (H, W, 3)

    # ---- Save config ----
    config = dict(
        model="convmixer", h=h, depth=depth, patch_size=patch_size,
        kernel_size=kernel_size, lr=lr, epochs=epochs,
        batch_size=batch_size, patience=patience,
        input_shape=list(input_shape), n_classes=n_classes,
    )
    with open(cfg_path, "w") as f:
        json.dump(config, f, indent=2)

    # ---- Build model ----
    model = build_convmixer(
        input_shape=input_shape,
        h=h, depth=depth,
        patch_size=patch_size,
        kernel_size=kernel_size,
        n_classes=n_classes,
    )

    if verbose:
        model.summary()

    # ---- LR schedule: cosine decay ----
    lr_schedule = keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=lr,
        decay_steps=epochs * (len(X_train) // batch_size),
        alpha=1e-6,
    )

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr_schedule),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    # ---- Keras callbacks ----
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=patience,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=ckpt_path,
            monitor="val_loss",
            save_best_only=True,
            verbose=0,
        ),
        keras.callbacks.CSVLogger(log_path, append=False),
    ]

    # ---- Train ----
    print(f"\nTraining ConvMixer  h={h}  depth={depth}  "
          f"patch={patch_size}  kernel={kernel_size}")
    print(f"Logs → {log_path}\n")

    hist = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1 if verbose else 0,
    )

    # ---- Test evaluation ----
    print("\nEvaluating on test set...")
    test_loss, test_acc = model.evaluate(X_test, y_test,
                                         batch_size=batch_size, verbose=0)
    print(f"\n{'='*40}")
    print(f"Test Loss:     {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"{'='*40}")

    # ---- Build history dict matching other models' format ----
    n_epochs_run = len(hist.history["loss"])
    history = {
        "epoch":        list(range(1, n_epochs_run + 1)),
        "train_loss":   [round(v, 6) for v in hist.history["loss"]],
        "val_loss":     [round(v, 6) for v in hist.history["val_loss"]],
        "train_acc":    [round(v, 4) for v in hist.history["accuracy"]],
        "val_acc":      [round(v, 4) for v in hist.history["val_accuracy"]],
        "learning_rate": [lr] * n_epochs_run,   # schedule handled internally
    }

    # Save test result alongside log
    result_path = os.path.join(log_dir, f"{run_name}_test_result.json")
    with open(result_path, "w") as f:
        json.dump({"test_loss": test_loss, "test_acc": test_acc}, f, indent=2)

    return history, model


# ===========================================================================
# CLI
# ===========================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train ConvMixer.")
    parser.add_argument("--h",             type=int,   default=256)
    parser.add_argument("--depth",         type=int,   default=8)
    parser.add_argument("--patch-size",    type=int,   default=7)
    parser.add_argument("--kernel-size",   type=int,   default=9)
    parser.add_argument("--lr",            type=float, default=1e-3)
    parser.add_argument("--epochs",        type=int,   default=50)
    parser.add_argument("--batch-size",    type=int,   default=64)
    parser.add_argument("--patience",      type=int,   default=10)
    parser.add_argument("--processed-dir", default="data/processed")
    args = parser.parse_args()

    history, model = train_convmixer(
        processed_dir=args.processed_dir,
        h=args.h,
        depth=args.depth,
        patch_size=args.patch_size,
        kernel_size=args.kernel_size,
        lr=args.lr,
        epochs=args.epochs,
        batch_size=args.batch_size,
        patience=args.patience,
    )