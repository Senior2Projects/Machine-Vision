"""
preprocessing.py
----------------
Resize every image to a fixed spatial size and normalize pixel values
using the minicv library. Reads from data/splits/{split}.csv,
processes each image, and saves results to data/processed/{split}.npz

Output per split:
    images : float32 array  (N, H, W, 3)  — normalized to [0, 1]
    labels : int32 array    (N,)           — integer class indices
    paths  : str array      (N,)           — original file paths

Usage:
    python preprocessing.py               # default 64x64
    python preprocessing.py --size 96     # override size
"""

import os
import csv
import argparse
import numpy as np
import sys

# ---------------------------------------------------------------------------
# Adjust this import path if minicv lives elsewhere relative to this script
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import minicv

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
TARGET_H = 64
TARGET_W = 64
SPLITS    = ["train", "val", "test"]
SPLITS_DIR    = os.path.join("data", "splits")    # where train/val/test.csv live
PROCESSED_DIR = os.path.join("data", "processed") # output .npz files go here

# Maps class name strings → integer indices (sorted for reproducibility)
CLASS_NAMES = sorted(["buildings", "forest", "glacier", "mountain", "sea", "street"])
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASS_NAMES)}


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

def load_and_preprocess(img_path: str, h: int, w: int) -> np.ndarray:
    """
    Load one image from disk, resize to (h, w), normalize to [0, 1].

    Parameters
    ----------
    img_path : str   Path to image file.
    h, w     : int   Target height and width.

    Returns
    -------
    np.ndarray  float32 array of shape (h, w, 3), values in [0, 1].

    Raises
    ------
    RuntimeError  If the image cannot be read or has unexpected shape.
    """
    img = minicv.read_image(img_path, mode="rgb")          # (H, W, 3) uint8

    if img.ndim != 3 or img.shape[2] != 3:
        raise RuntimeError(
            f"Expected RGB image at '{img_path}', got shape {img.shape}."
        )

    resized = minicv.resize(img, h, w, method="bilinear")  # float64 (h, w, 3)
    normed  = minicv.normalize(resized, mode="minmax")      # float64 [0, 1]

    return normed.astype(np.float32)


def process_split(split: str, h: int, w: int) -> None:
    """
    Process one split (train / val / test) end-to-end.

    Reads   : data/splits/{split}.csv  (columns: path, label)
    Writes  : data/processed/{split}.npz

    Parameters
    ----------
    split : str   One of 'train', 'val', 'test'.
    h, w  : int   Target spatial size.
    """
    csv_path = os.path.join(SPLITS_DIR, f"{split}.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Split file not found: '{csv_path}'")

    # Read annotation rows
    rows = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    n = len(rows)
    print(f"[{split}]  {n} images  →  target size ({h}, {w})")

    images = np.zeros((n, h, w, 3), dtype=np.float32)
    labels = np.zeros(n, dtype=np.int32)
    paths  = np.empty(n, dtype=object)

    failed = []
    for i, row in enumerate(rows):
        img_path  = os.path.normpath(row["filepath"])  # fix mixed separators
        label_str = row["label"]

        try:
            images[i] = load_and_preprocess(img_path, h, w)
            labels[i] = CLASS_TO_IDX[label_str]
            paths[i]  = img_path
        except Exception as e:
            print(f"  WARNING: skipping '{img_path}': {e}")
            failed.append(i)

    # Remove failed rows
    if failed:
        keep = [i for i in range(n) if i not in set(failed)]
        images = images[keep]
        labels = labels[keep]
        paths  = paths[keep]
        print(f"  {len(failed)} images skipped, {len(keep)} saved.")

    # Save
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    out_path = os.path.join(PROCESSED_DIR, f"{split}.npz")
    np.savez_compressed(out_path, images=images, labels=labels, paths=paths)
    print(f"  Saved → {out_path}  (images: {images.shape}, labels: {labels.shape})\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Preprocess Intel Image Classification splits.")
    parser.add_argument("--size", type=int, default=TARGET_H,
                        help=f"Square target size (default: {TARGET_H})")
    parser.add_argument("--splits", nargs="+", default=SPLITS,
                        choices=["train", "val", "test"],
                        help="Which splits to process (default: all)")
    args = parser.parse_args()

    h = w = args.size
    print(f"Preprocessing  size=({h}, {w})  splits={args.splits}\n")

    for split in args.splits:
        process_split(split, h, w)

    print("Done. All splits saved to data/processed/")


if __name__ == "__main__":
    main()