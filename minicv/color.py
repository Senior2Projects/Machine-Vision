import numpy as np

from .utils import validate_image


# ---------------------------------------------------------------------------
# RGB -> Grayscale
# ---------------------------------------------------------------------------

def rgb_to_gray(image):
    """
    Convert an RGB image to grayscale using the luminance formula.

    Parameters
    ----------
    image : np.ndarray
        RGB image of shape (H, W, 3). Values can be uint8 [0, 255]
        or float [0, 1].

    Returns
    -------
    np.ndarray
        Grayscale image of shape (H, W) with same dtype as input.

    Raises
    ------
    TypeError
        If image is not a NumPy ndarray.
    ValueError
        If image is not 3D or does not have exactly 3 channels.

    Notes
    -----
    Uses the ITU-R BT.601 luminance formula:
        Y = 0.2989 * R + 0.5870 * G + 0.1140 * B
    This matches the human eye's sensitivity to each color channel.
    Output shape is always (H, W) — the channel dimension is removed.
    """
    validate_image(image)
    if image.ndim != 3:
        raise ValueError(f"rgb_to_gray expects a 3D image (H, W, 3), got shape {image.shape}.")
    if image.shape[2] != 3:
        raise ValueError(f"rgb_to_gray expects exactly 3 channels, got {image.shape[2]}.")

    dtype = image.dtype
    image_float = image.astype(np.float64)

    gray = (
        0.2989 * image_float[:, :, 0] +
        0.5870 * image_float[:, :, 1] +
        0.1140 * image_float[:, :, 2]
    )

    # Preserve original dtype
    if np.issubdtype(dtype, np.integer):
        return np.clip(gray, 0, 255).astype(dtype)
    return gray.astype(dtype)


# ---------------------------------------------------------------------------
# Grayscale -> RGB
# ---------------------------------------------------------------------------

def gray_to_rgb(image):
    """
    Convert a grayscale image to a 3-channel RGB image.

    Parameters
    ----------
    image : np.ndarray
        Grayscale image of shape (H, W). Values can be uint8 [0, 255]
        or float [0, 1].

    Returns
    -------
    np.ndarray
        RGB image of shape (H, W, 3) with same dtype as input.
        All three channels are identical (no color is added).

    Raises
    ------
    TypeError
        If image is not a NumPy ndarray.
    ValueError
        If image is not 2D.

    Notes
    -----
    This is a utility conversion used when an RGB-format output is needed
    but the source is grayscale. The result is a valid RGB array but
    visually still appears gray.
    """
    validate_image(image)
    if image.ndim != 2:
        raise ValueError(f"gray_to_rgb expects a 2D grayscale image, got shape {image.shape}.")

    return np.stack([image, image, image], axis=-1)