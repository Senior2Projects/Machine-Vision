"""
extractor.py
------------
Feature extraction pipeline for Milestone 2.
Extracts 3 feature families per image using minicv descriptors,
concatenates into one vector per image.

Feature Vector Layout (total = 582 dimensions at default settings)
-------------------------------------------------------------------
Family 1 — Color (global)        indices [0 : 96]
    color_histogram_descriptor   bins=32 × 3 channels = 96 dims

Family 2 — Shape (global)        indices [96 : 103]
    hu_moments_descriptor        = 7 dims

Family 3 — Texture / Gradient    indices [103 : 582]
    hog_descriptor               cell_size=10, bins=9
                                 n_cells = (150//10)^2 = 225 cells
                                 225 × 9 = 2025 dims  → [103 : 2128]

    lbp_descriptor               radius=1, n_points=8, bins=256
                                 = 256 dims             → [2128 : 2384]

    *** total = 96 + 7 + 2025 + 256 = 2384 dims ***

    Note: actual HOG dims depend on image size. At 150×150 with
    cell_size=10 → 15×15 cells → 2025 dims.
    At 96×96 → 9×9 cells (after crop to 90) → 729 dims.
    The index scheme is fixed at runtime and printed on first call.

Usage
-----
    from features.extractor import extract_features, extract_dataset

    # single image  float32 (H, W, 3) in [0, 1]
    vec = extract_features(image)           # 1D np.ndarray

    # whole split
    X = extract_dataset(images, verbose=True)   # (N, D)
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import minicv

# ---------------------------------------------------------------------------
# Descriptor hyperparameters — change here only
# ---------------------------------------------------------------------------
COLOR_HIST_BINS = 32        # per channel → ×3 for RGB
HOG_CELL_SIZE   = 10        # pixels per cell (both axes)
HOG_BINS        = 9         # orientation bins
LBP_RADIUS      = 1
LBP_POINTS      = 8
LBP_BINS        = 256


# ---------------------------------------------------------------------------
# Index registry — built once on first call to extract_features
# ---------------------------------------------------------------------------
_INDEX_REGISTRY: dict | None = None


def _build_index_registry(image: np.ndarray) -> dict:
    """
    Run descriptors on a dummy image to compute exact dimensions,
    then build and return the index registry.
    """
    # Color histogram
    color_vec = minicv.color_histogram_descriptor(image, bins=COLOR_HIST_BINS)
    n_color   = len(color_vec)

    # Hu moments
    hu_vec  = minicv.hu_moments_descriptor(image)
    n_hu    = len(hu_vec)

    # HOG
    hog_vec = minicv.hog_descriptor(image, cell_size=HOG_CELL_SIZE, bins=HOG_BINS)
    n_hog   = len(hog_vec)

    # LBP
    lbp_vec = minicv.lbp_descriptor(image, radius=LBP_RADIUS,
                                     n_points=LBP_POINTS, bins=LBP_BINS)
    n_lbp   = len(lbp_vec)

    total = n_color + n_hu + n_hog + n_lbp

    registry = {
        "color_hist": {
            "start": 0,
            "end":   n_color,
            "dims":  n_color,
            "desc":  f"Color histogram  (bins={COLOR_HIST_BINS} × 3 channels)",
        },
        "hu_moments": {
            "start": n_color,
            "end":   n_color + n_hu,
            "dims":  n_hu,
            "desc":  "Hu moment invariants (7 values, log-transformed)",
        },
        "hog": {
            "start": n_color + n_hu,
            "end":   n_color + n_hu + n_hog,
            "dims":  n_hog,
            "desc":  (f"HOG  (cell_size={HOG_CELL_SIZE}, bins={HOG_BINS}, "
                      f"n_cells={n_hog // HOG_BINS})"),
        },
        "lbp": {
            "start": n_color + n_hu + n_hog,
            "end":   n_color + n_hu + n_hog + n_lbp,
            "dims":  n_lbp,
            "desc":  (f"LBP  (radius={LBP_RADIUS}, n_points={LBP_POINTS}, "
                      f"bins={LBP_BINS})"),
        },
        "total": total,
    }
    return registry


def get_index_registry(sample_image: np.ndarray | None = None) -> dict:
    """
    Return the feature index registry (build it on first call).

    Parameters
    ----------
    sample_image : np.ndarray | None
        Any image from the dataset (float32 H×W×3). Required on first call.
        Subsequent calls return the cached registry without needing an image.

    Returns
    -------
    dict  with keys: color_hist, hu_moments, hog, lbp, total
          each containing: start, end, dims, desc
    """
    global _INDEX_REGISTRY
    if _INDEX_REGISTRY is None:
        if sample_image is None:
            raise RuntimeError(
                "Index registry not yet built. Pass a sample_image on first call."
            )
        _INDEX_REGISTRY = _build_index_registry(sample_image)
    return _INDEX_REGISTRY


def print_index_registry(registry: dict) -> None:
    """Print a human-readable feature index table."""
    print("\n" + "=" * 62)
    print(f"  Feature Vector Index Scheme   (total = {registry['total']} dims)")
    print("=" * 62)
    for key in ("color_hist", "hu_moments", "hog", "lbp"):
        r = registry[key]
        print(f"  [{r['start']:>5} : {r['end']:>5}]  {r['desc']}")
    print("=" * 62 + "\n")


# ---------------------------------------------------------------------------
# Core extraction
# ---------------------------------------------------------------------------

def extract_features(image: np.ndarray) -> np.ndarray:
    """
    Extract and concatenate all feature families from one image.

    Parameters
    ----------
    image : np.ndarray
        Float32 array of shape (H, W, 3), values in [0, 1].
        Must be RGB (3-channel).

    Returns
    -------
    np.ndarray
        1D float32 feature vector of shape (D,) where D = total dims
        as defined in the index registry.

    Notes
    -----
    Extraction order (matches index registry):
        1. color_histogram_descriptor  — RGB color distribution
        2. hu_moments_descriptor       — shape, rotation/scale invariant
        3. hog_descriptor              — local gradient structure
        4. lbp_descriptor              — local texture patterns

    Input is scaled to [0, 255] uint8 range before passing to minicv
    descriptors that expect uint8 (color_hist, hu_moments).
    HOG and LBP work on the float image directly via rgb_to_gray internally.
    """
    # minicv descriptors expect values in [0, 255] range
    img_255 = (image * 255.0).astype(np.float64)

    color_vec = minicv.color_histogram_descriptor(img_255, bins=COLOR_HIST_BINS)
    hu_vec    = minicv.hu_moments_descriptor(img_255)
    hog_vec   = minicv.hog_descriptor(img_255, cell_size=HOG_CELL_SIZE, bins=HOG_BINS)
    lbp_vec   = minicv.lbp_descriptor(img_255, radius=LBP_RADIUS,
                                       n_points=LBP_POINTS, bins=LBP_BINS)

    vec = np.concatenate([color_vec, hu_vec, hog_vec, lbp_vec])
    return vec.astype(np.float32)


def extract_dataset(
    images: np.ndarray,
    verbose: bool = True,
    log_every: int = 200,
) -> np.ndarray:
    """
    Extract features for an entire image dataset.

    Parameters
    ----------
    images    : np.ndarray  float32 (N, H, W, 3)
    verbose   : bool        Print progress every log_every images.
    log_every : int         Progress print interval.

    Returns
    -------
    np.ndarray  float32 (N, D)  — one feature vector per image.

    Notes
    -----
    The index registry is built from the first image and cached globally.
    All subsequent images must have the same spatial dimensions.
    """
    N = len(images)

    # Build registry from first image and print it
    registry = get_index_registry(images[0])
    if verbose:
        print_index_registry(registry)
        print(f"Extracting features for {N} images...")

    D = registry["total"]
    X = np.zeros((N, D), dtype=np.float32)

    for i in range(N):
        X[i] = extract_features(images[i])
        if verbose and (i + 1) % log_every == 0:
            print(f"  {i + 1}/{N} done")

    if verbose:
        print(f"  Done. Feature matrix shape: {X.shape}")

    return X


# ---------------------------------------------------------------------------
# Convenience: extract and save all splits
# ---------------------------------------------------------------------------

def extract_and_save_all(
    processed_dir: str | None = None,
    features_dir:  str | None = None,
    use_augmented: bool = True,
    verbose: bool = True,
) -> None:
    """
    Load preprocessed .npz splits, extract features, save to features_dir.

    Reads
    -----
    data/processed/train_augmented.npz  (or train.npz if use_augmented=False)
    data/processed/val.npz
    data/processed/test.npz

    Writes
    ------
    features/train_features.npz   — {X: (N,D), y: (N,), registry: dict}
    features/val_features.npz
    features/test_features.npz

    Parameters
    ----------
    processed_dir : str   Path to data/processed/. Auto-detected if None.
    features_dir  : str   Path to features/. Auto-detected if None.
    use_augmented : bool  Use train_augmented.npz instead of train.npz.
    verbose       : bool  Print progress.
    """
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    if processed_dir is None:
        processed_dir = os.path.join(root, "data", "processed")
    if features_dir is None:
        features_dir = os.path.join(root, "features")

    os.makedirs(features_dir, exist_ok=True)

    splits = {
        "train": "train_augmented.npz" if use_augmented else "train.npz",
        "val":   "val.npz",
        "test":  "test.npz",
    }

    for split, fname in splits.items():
        fpath = os.path.join(processed_dir, fname)
        if not os.path.exists(fpath):
            print(f"  WARNING: '{fpath}' not found, skipping {split}.")
            continue

        print(f"\n[{split}] Loading {fname}...")
        data   = np.load(fpath, allow_pickle=True)
        images = data["images"]   # float32 (N, H, W, 3)
        labels = data["labels"]   # int32   (N,)

        X = extract_dataset(images, verbose=verbose)

        out_path = os.path.join(features_dir, f"{split}_features.npz")
        np.savez_compressed(out_path, X=X, y=labels)
        print(f"  Saved → {out_path}  X={X.shape}  y={labels.shape}")

    # Save registry separately for later reference
    registry = get_index_registry()
    reg_path = os.path.join(features_dir, "feature_index.npz")
    np.savez(reg_path, registry=np.array([registry], dtype=object))
    print(f"\nIndex registry saved → {reg_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extract features for all splits.")
    parser.add_argument("--no-augmented", action="store_true",
                        help="Use train.npz instead of train_augmented.npz")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    extract_and_save_all(
        use_augmented=not args.no_augmented,
        verbose=not args.quiet,
    )
