import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os

from .utils import validate_image


# ---------------------------------------------------------------------------
# Read
# ---------------------------------------------------------------------------

def read_image(path, mode="rgb"):
    """
    Load an image from disk into a NumPy array.

    Parameters
    ----------
    path : str
        Path to the image file. Supported formats: PNG, JPG (depends on
        Matplotlib backend).
    mode : str
        Output color mode. One of:
        - 'rgb'  : returns image as (H, W, 3) uint8 array.
        - 'gray' : returns image as (H, W) uint8 array.

    Returns
    -------
    np.ndarray
        Image array with dtype uint8.
        Shape (H, W, 3) for RGB or (H, W) for grayscale.

    Raises
    ------
    TypeError
        If path is not a string, or mode is not a string.
    ValueError
        If path does not exist, file extension is unsupported,
        or mode is not 'rgb' or 'gray'.

    Notes
    -----
    Matplotlib reads PNG as float32 in [0, 1] and JPG as uint8 in [0, 255].
    This function normalizes both to uint8 [0, 255] for consistency.
    """
    if not isinstance(path, str):
        raise TypeError(f"path must be a string, got {type(path).__name__}.")
    if not isinstance(mode, str):
        raise TypeError(f"mode must be a string, got {type(mode).__name__}.")
    if not os.path.exists(path):
        raise ValueError(f"File not found: '{path}'.")

    ext = os.path.splitext(path)[-1].lower()
    if ext not in (".png", ".jpg", ".jpeg"):
        raise ValueError(f"Unsupported file format '{ext}'. Use PNG or JPG.")

    if mode not in ("rgb", "gray"):
        raise ValueError(f"mode must be 'rgb' or 'gray', got '{mode}'.")

    image = mpimg.imread(path)

    # Normalize to uint8
    if image.dtype == np.float32 or image.dtype == np.float64:
        image = (image * 255).clip(0, 255).astype(np.uint8)

    # Drop alpha channel if present (RGBA -> RGB)
    if image.ndim == 3 and image.shape[2] == 4:
        image = image[:, :, :3]

    # Handle grayscale PNG loaded as (H, W) already
    if image.ndim == 2 and mode == "rgb":
        image = np.stack([image, image, image], axis=-1)

    if mode == "gray":
        if image.ndim == 3:
            # Import here to avoid circular import
            from .color import rgb_to_gray
            image = rgb_to_gray(image)

    return image


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

def save_image(image, path):
    """
    Save a NumPy image array to disk as PNG or JPG.

    Parameters
    ----------
    image : np.ndarray
        Image array to save. Accepts:
        - (H, W)    grayscale
        - (H, W, 3) RGB
    path : str
        Destination file path including extension (.png or .jpg/.jpeg).

    Returns
    -------
    None

    Raises
    ------
    TypeError
        If image is not a NumPy ndarray, or path is not a string.
    ValueError
        If image shape is not supported, or file extension is unsupported.

    Notes
    -----
    If image dtype is float, values are expected in [0, 1] and will be
    scaled to [0, 255] before saving. uint8 arrays are saved as-is.
    """
    validate_image(image)
    if not isinstance(path, str):
        raise TypeError(f"path must be a string, got {type(path).__name__}.")

    ext = os.path.splitext(path)[-1].lower()
    if ext not in (".png", ".jpg", ".jpeg"):
        raise ValueError(f"Unsupported file format '{ext}'. Use PNG or JPG.")

    if image.ndim == 3 and image.shape[2] != 3:
        raise ValueError(f"RGB image must have 3 channels, got {image.shape[2]}.")

    # Convert float to uint8 if needed
    if np.issubdtype(image.dtype, np.floating):
        image = (image * 255).clip(0, 255).astype(np.uint8)

    # Ensure output directory exists
    out_dir = os.path.dirname(path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir)

    plt.imsave(path, image, cmap="gray" if image.ndim == 2 else None)