"""
mrmr_selection.py
-----------------
MRMR (Minimum Redundancy Maximum Relevance) feature selection.
Applied to the training feature matrix to select the top-K most
informative, non-redundant features.

Library used: mrmr-selection (pip install mrmr-selection)
All feature extraction is still done via extractor.py (minicv-based).

Usage
-----
    from features.mrmr_selection import run_mrmr, load_selected_features

    # Select top 200 features from training set
    selected_indices = run_mrmr(X_train, y_train, K=200)

    # Apply selection to all splits
    X_train_sel = X_train[:, selected_indices]
    X_val_sel   = X_val[:,   selected_indices]
    X_test_sel  = X_test[:,  selected_indices]
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ---------------------------------------------------------------------------
# MRMR selection
# ---------------------------------------------------------------------------

def run_mrmr(
    X_train: np.ndarray,
    y_train: np.ndarray,
    K: int = 200,
    verbose: bool = True,
) -> np.ndarray:
    """
    Run MRMR feature selection on the training set.

    Parameters
    ----------
    X_train : np.ndarray  float32 (N, D)  — training feature matrix
    y_train : np.ndarray  int32   (N,)    — class labels
    K       : int         Number of top features to select.
    verbose : bool        Print selected feature count and indices summary.

    Returns
    -------
    selected_indices : np.ndarray  int64 (K,)
        Indices into the feature vector, ordered by MRMR relevance score.
        Apply as: X_selected = X[:, selected_indices]

    Notes
    -----
    Uses the mrmr-selection library (pip install mrmr-selection).
    MRMR selects features that maximize relevance to the target label
    while minimizing redundancy among selected features.

    The selection is run only on X_train — val and test sets then have
    the same column indices applied without any label leakage.
    """
    try:
        import pandas as pd
        from mrmr import mrmr_classif
    except ImportError:
        raise ImportError(
            "mrmr-selection is not installed. Run:\n"
            "    pip install mrmr-selection"
        )

    if K > X_train.shape[1]:
        raise ValueError(
            f"K={K} exceeds feature dimensionality {X_train.shape[1]}. "
            f"Choose K <= {X_train.shape[1]}."
        )

    if verbose:
        print(f"Running MRMR: selecting top {K} from {X_train.shape[1]} features "
              f"using {X_train.shape[0]} training samples...")

    # mrmr_classif expects a pandas DataFrame
    col_names = [f"f{i:04d}" for i in range(X_train.shape[1])]
    df_X = pd.DataFrame(X_train, columns=col_names)
    sr_y = pd.Series(y_train, name="label")

    selected_names = mrmr_classif(X=df_X, y=sr_y, K=K)

    # Convert column names back to integer indices
    name_to_idx = {name: i for i, name in enumerate(col_names)}
    selected_indices = np.array([name_to_idx[n] for n in selected_names], dtype=np.int64)

    if verbose:
        print(f"  Selected {len(selected_indices)} features.")
        print(f"  Index range: {selected_indices.min()} – {selected_indices.max()}")

    return selected_indices


# ---------------------------------------------------------------------------
# Apply selection and save
# ---------------------------------------------------------------------------

def apply_and_save(
    selected_indices: np.ndarray,
    features_dir: str | None = None,
    K: int | None = None,
    verbose: bool = True,
) -> dict:
    """
    Load raw feature splits, apply MRMR selection, save reduced splits.

    Reads
    -----
    features/train_features.npz
    features/val_features.npz
    features/test_features.npz

    Writes
    ------
    features/train_mrmr.npz
    features/val_mrmr.npz
    features/test_mrmr.npz
    features/mrmr_indices.npy   — the selected_indices array

    Parameters
    ----------
    selected_indices : np.ndarray  int64 (K,)
    features_dir     : str | None  auto-detected if None
    K                : int | None  used only for display
    verbose          : bool

    Returns
    -------
    dict with keys 'train', 'val', 'test' each containing (X_sel, y)
    """
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if features_dir is None:
        features_dir = os.path.join(root, "features")

    result = {}
    for split in ("train", "val", "test"):
        fpath = os.path.join(features_dir, f"{split}_features.npz")
        if not os.path.exists(fpath):
            print(f"  WARNING: '{fpath}' not found, skipping.")
            continue

        data = np.load(fpath)
        X    = data["X"]
        y    = data["y"]

        X_sel = X[:, selected_indices]
        result[split] = (X_sel, y)

        out_path = os.path.join(features_dir, f"{split}_mrmr.npz")
        np.savez_compressed(out_path, X=X_sel, y=y)

        if verbose:
            print(f"  [{split}]  {X.shape} → {X_sel.shape}  saved → {out_path}")

    # Save indices for reproducibility
    idx_path = os.path.join(features_dir, "mrmr_indices.npy")
    np.save(idx_path, selected_indices)
    if verbose:
        print(f"  MRMR indices saved → {idx_path}")

    return result


# ---------------------------------------------------------------------------
# One-shot convenience
# ---------------------------------------------------------------------------

def run_and_save(
    K: int = 200,
    features_dir: str | None = None,
    verbose: bool = True,
) -> np.ndarray:
    """
    Load train features, run MRMR, apply to all splits, save everything.

    Parameters
    ----------
    K            : int   Number of features to select.
    features_dir : str   Path to features/. Auto-detected if None.
    verbose      : bool

    Returns
    -------
    selected_indices : np.ndarray int64 (K,)
    """
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if features_dir is None:
        features_dir = os.path.join(root, "features")

    train_path = os.path.join(features_dir, "train_features.npz")
    if not os.path.exists(train_path):
        raise FileNotFoundError(
            f"'{train_path}' not found. Run extractor.py first."
        )

    data    = np.load(train_path)
    X_train = data["X"]
    y_train = data["y"]

    selected_indices = run_mrmr(X_train, y_train, K=K, verbose=verbose)
    apply_and_save(selected_indices, features_dir=features_dir, K=K, verbose=verbose)

    return selected_indices


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run MRMR feature selection.")
    parser.add_argument("--K", type=int, default=200,
                        help="Number of features to select (default: 200)")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    run_and_save(K=args.K, verbose=not args.quiet)
