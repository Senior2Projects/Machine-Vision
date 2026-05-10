import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import minicv


# ---------------------------------------------------------------------------
# Constants — tweak these without touching logic
# ---------------------------------------------------------------------------
MAX_ANGLE      = 15       # degrees
MAX_SHIFT_FRAC = 0.10     # fraction of image size used for translate range
BRIGHTNESS_LO  = 0.75     # lower bound of brightness multiplier
BRIGHTNESS_HI  = 1.25     # upper bound of brightness multiplier
NOISE_STD      = 0.04     # std of Gaussian noise (image in [0,1])
CROP_FRAC_LO   = 0.80     # minimum crop window fraction
CROP_FRAC_HI   = 0.95     # maximum crop window fraction
GAUSS_SIZE     = 5        # kernel size for gaussian_noise smoothing (pre-add)
GAUSS_SIGMA    = 1.0


# ---------------------------------------------------------------------------
# Individual transforms
# Each function:
#   input  — np.ndarray float32 (H, W, 3) in [0, 1]
#   output — np.ndarray float32 (H, W, 3) in [0, 1]
# ---------------------------------------------------------------------------

def horizontal_flip(image: np.ndarray) -> np.ndarray:
    """
    Flip image horizontally (left ↔ right).

    Uses NumPy slice reversal — no minicv call needed since this is
    a pure array operation with no image-processing logic.
    """
    return image[:, ::-1, :].copy()


def random_rotate(image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """
    Rotate image by a random angle in [-MAX_ANGLE, +MAX_ANGLE] degrees.

    Uses minicv.rotate with bilinear interpolation.
    Black pixels fill areas outside the original canvas boundary.
    """
    angle = rng.uniform(-MAX_ANGLE, MAX_ANGLE)
    rotated = minicv.rotate(image, angle=angle, method="bilinear")  # float64
    return np.clip(rotated, 0.0, 1.0).astype(np.float32)


def random_translate(image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """
    Translate image by a random integer pixel offset.

    Shift range is ±(MAX_SHIFT_FRAC × image_size) in each axis.
    Uses minicv.translate which requires integer shifts.
    """
    H, W = image.shape[:2]
    max_dy = max(1, int(H * MAX_SHIFT_FRAC))
    max_dx = max(1, int(W * MAX_SHIFT_FRAC))

    shift_x = int(rng.integers(-max_dx, max_dx + 1))
    shift_y = int(rng.integers(-max_dy, max_dy + 1))

    translated = minicv.translate(image, shift_x=shift_x, shift_y=shift_y)  # float64
    return np.clip(translated, 0.0, 1.0).astype(np.float32)


def brightness_jitter(image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """
    Multiply pixel values by a random scalar per channel.

    Each of R, G, B is scaled independently by a factor drawn from
    [BRIGHTNESS_LO, BRIGHTNESS_HI]. Uses minicv.normalize to remap
    the result back to [0, 1] after scaling.

    Note: independent per-channel scaling also introduces mild hue shifts,
    simulating real-world illumination variation.
    """
    factors = rng.uniform(BRIGHTNESS_LO, BRIGHTNESS_HI, size=(3,))
    jittered = image.astype(np.float64) * factors[np.newaxis, np.newaxis, :]

    # minicv.normalize mode='minmax' → [0, 1]
    normed = minicv.normalize(jittered, mode="minmax")
    return normed.astype(np.float32)


def gaussian_noise(image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """
    Add zero-mean Gaussian noise to the image.

    Noise is generated with std=NOISE_STD and clipped to [0, 1].
    Applied per-channel independently. A mild Gaussian blur
    (via minicv.gaussian_filter) is applied to the noise map first
    to produce spatially correlated noise — closer to real sensor noise.
    """
    H, W, C = image.shape
    noise_raw = rng.normal(0.0, NOISE_STD, size=(H, W, C)).astype(np.float32)

    # Smooth the noise slightly per channel using minicv.gaussian_filter
    noise_smooth = np.stack([
        minicv.gaussian_filter(
            noise_raw[:, :, c].astype(np.float64),
            kernel_size=GAUSS_SIZE, sigma=GAUSS_SIGMA
        ).astype(np.float32)
        for c in range(C)
    ], axis=-1)

    noisy = image + noise_smooth
    return np.clip(noisy, 0.0, 1.0).astype(np.float32)


def random_crop_resize(image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """
    Crop a random sub-window then resize back to the original spatial size.

    Crop fraction is drawn uniformly from [CROP_FRAC_LO, CROP_FRAC_HI].
    The crop origin is drawn so the crop window stays within the image.
    Resize uses minicv.resize with bilinear interpolation.

    This simulates zoom-in and slight viewpoint changes.
    """
    H, W = image.shape[:2]
    frac   = rng.uniform(CROP_FRAC_LO, CROP_FRAC_HI)
    crop_h = max(1, int(H * frac))
    crop_w = max(1, int(W * frac))

    top  = int(rng.integers(0, H - crop_h + 1))
    left = int(rng.integers(0, W - crop_w + 1))

    cropped = image[top:top + crop_h, left:left + crop_w, :]  # (crop_h, crop_w, 3)
    resized = minicv.resize(cropped, H, W, method="bilinear")  # float64
    return np.clip(resized, 0.0, 1.0).astype(np.float32)


# ---------------------------------------------------------------------------
# Transform registry — ordered list used by augment_image
# Each entry: (name, fn)  where fn(image, rng) → image
# horizontal_flip doesn't need rng but we pass it for uniform signature
# ---------------------------------------------------------------------------
_TRANSFORMS = [
    ("horizontal_flip",   lambda img, rng: horizontal_flip(img)),
    ("random_rotate",     random_rotate),
    ("random_translate",  random_translate),
    ("brightness_jitter", brightness_jitter),
    ("gaussian_noise",    gaussian_noise),
    ("random_crop_resize",random_crop_resize),
]

TRANSFORM_NAMES = [name for name, _ in _TRANSFORMS]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def augment_image(
    image: np.ndarray,
    label: int,
    rng: np.random.Generator,
    transforms: list[str] | None = None,
) -> tuple[np.ndarray, int]:
    """
    Apply a random subset of augmentation transforms to a single image.

    One transform is selected at random from the available pool and applied.
    This keeps each augmented copy distinct while limiting cascade distortion.

    Parameters
    ----------
    image      : np.ndarray float32 (H, W, 3) in [0, 1]
    label      : int   class index — passed through unchanged
    rng        : np.random.Generator   seeded RNG for reproducibility
    transforms : list[str] | None
        Names of transforms to sample from. If None, uses all 6.
        Valid names: horizontal_flip, random_rotate, random_translate,
                     brightness_jitter, gaussian_noise, random_crop_resize

    Returns
    -------
    aug_image  : np.ndarray float32 (H, W, 3) in [0, 1]
    label      : int  (unchanged)

    Raises
    ------
    ValueError  If an unknown transform name is requested.
    """
    pool = _TRANSFORMS
    if transforms is not None:
        valid = {name for name, _ in _TRANSFORMS}
        for t in transforms:
            if t not in valid:
                raise ValueError(f"Unknown transform '{t}'. Valid: {sorted(valid)}")
        pool = [(name, fn) for name, fn in _TRANSFORMS if name in transforms]

    name, fn = pool[int(rng.integers(0, len(pool)))]
    aug = fn(image, rng)
    return aug, label


def augment_dataset(
    images: np.ndarray,
    labels: np.ndarray,
    multiplier: int = 3,
    seed: int = 42,
    transforms: list[str] | None = None,
    verbose: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Augment an entire dataset by generating additional copies of each image.

    Only generates NEW augmented copies — does NOT include the originals.
    Caller decides whether to concatenate originals + augmented.

    Parameters
    ----------
    images     : np.ndarray  float32 (N, H, W, 3)
    labels     : np.ndarray  int32   (N,)
    multiplier : int   Number of augmented copies per original (default 3).
    seed       : int   RNG seed for reproducibility (default 42).
    transforms : list[str] | None   Subset of transform names to use.
    verbose    : bool  Print progress every 500 images.

    Returns
    -------
    aug_images : np.ndarray  float32 (N * multiplier, H, W, 3)
    aug_labels : np.ndarray  int32   (N * multiplier,)

    Notes
    -----
    With multiplier=3 and 700 train images per class (6 classes = 4200 total):
        augmented pool = 4200 × 3 = 12600 additional images
        combined train = 4200 + 12600 = 16800 images
    """
    if multiplier < 1:
        raise ValueError(f"multiplier must be >= 1, got {multiplier}.")

    N, H, W, C = images.shape
    total = N * multiplier
    rng   = np.random.default_rng(seed)

    aug_images = np.zeros((total, H, W, C), dtype=np.float32)
    aug_labels = np.zeros(total, dtype=np.int32)

    idx = 0
    for i in range(N):
        for _ in range(multiplier):
            aug_images[idx], aug_labels[idx] = augment_image(
                images[i], labels[i], rng, transforms
            )
            idx += 1
        if verbose and (i + 1) % 500 == 0:
            print(f"  augmented {i + 1}/{N} originals  ({idx} copies so far)")

    if verbose:
        print(f"  Done — {total} augmented images generated.")

    return aug_images, aug_labels


def load_and_augment_train(
    processed_dir: str = os.path.join("data", "processed"),
    multiplier: int = 3,
    seed: int = 42,
    save: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convenience function: load train.npz, augment, optionally save result.

    Saves to data/processed/train_augmented.npz so the feature extraction
    pipeline can load it without re-running augmentation.

    Parameters
    ----------
    processed_dir : str   Path to data/processed/ directory.
    multiplier    : int   Augmented copies per original image.
    seed          : int   RNG seed.
    save          : bool  Write train_augmented.npz to disk.

    Returns
    -------
    combined_images : float32 (N_orig + N_aug, H, W, 3)
    combined_labels : int32   (N_orig + N_aug,)
    """
    train_path = os.path.join(processed_dir, "train.npz")
    if not os.path.exists(train_path):
        raise FileNotFoundError(
            f"Preprocessed train split not found at '{train_path}'. "
            "Run preprocessing.py first."
        )

    data   = np.load(train_path, allow_pickle=True)
    images = data["images"]    # float32 (N, H, W, 3)
    labels = data["labels"]    # int32   (N,)

    print(f"Loaded train split: {images.shape[0]} images")
    print(f"Generating {multiplier}× augmented copies...")

    aug_images, aug_labels = augment_dataset(
        images, labels, multiplier=multiplier, seed=seed, verbose=True
    )

    combined_images = np.concatenate([images, aug_images], axis=0)
    combined_labels = np.concatenate([labels, aug_labels], axis=0)

    print(f"Combined train: {combined_images.shape[0]} images "
          f"(orig {images.shape[0]} + aug {aug_images.shape[0]})")

    if save:
        out_path = os.path.join(processed_dir, "train_augmented.npz")
        np.savez_compressed(
            out_path,
            images=combined_images,
            labels=combined_labels,
        )
        print(f"Saved → {out_path}")

    return combined_images, combined_labels


# ---------------------------------------------------------------------------
# CLI: run augmentation standalone
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Augment the training split.")
    parser.add_argument("--multiplier", type=int, default=3)
    parser.add_argument("--seed",       type=int, default=42)
    parser.add_argument("--no-save",    action="store_true")
    args = parser.parse_args()

    load_and_augment_train(
        multiplier=args.multiplier,
        seed=args.seed,
        save=not args.no_save,
    )
