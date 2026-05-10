import numpy as np


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_image(image, name="image"):
    """
    Validate that input is a proper NumPy image array.

    Parameters
    ----------
    image : np.ndarray
        Input image to validate.
    name : str
        Variable name used in error messages.

    Raises
    ------
    TypeError
        If image is not a NumPy ndarray.
    ValueError
        If image is not 2D (grayscale) or 3D (RGB).
    """
    if not isinstance(image, np.ndarray):
        raise TypeError(f"{name} must be a NumPy ndarray, got {type(image).__name__}.")
    if image.ndim not in (2, 3):
        raise ValueError(f"{name} must be 2D (grayscale) or 3D (RGB), got shape {image.shape}.")


def validate_kernel(kernel):
    """
    Validate a convolution kernel.

    Parameters
    ----------
    kernel : np.ndarray
        Kernel to validate.

    Raises
    ------
    TypeError
        If kernel is not a NumPy ndarray or contains non-numeric values.
    ValueError
        If kernel is empty, not 2D, or does not have odd dimensions.
    """
    if not isinstance(kernel, np.ndarray):
        raise TypeError(f"Kernel must be a NumPy ndarray, got {type(kernel).__name__}.")
    if kernel.ndim != 2:
        raise ValueError(f"Kernel must be 2D, got shape {kernel.shape}.")
    if kernel.size == 0:
        raise ValueError("Kernel must not be empty.")
    if kernel.shape[0] % 2 == 0 or kernel.shape[1] % 2 == 0:
        raise ValueError(f"Kernel dimensions must be odd, got {kernel.shape}.")
    if not np.issubdtype(kernel.dtype, np.number):
        raise TypeError("Kernel must contain numeric values.")


# ---------------------------------------------------------------------------
# Clip
# ---------------------------------------------------------------------------

def clip_pixels(image, low=0.0, high=255.0):
    """
    Clip pixel values to a specified range.

    Parameters
    ----------
    image : np.ndarray
        Input image array.
    low : float
        Minimum allowed pixel value. Default is 0.
    high : float
        Maximum allowed pixel value. Default is 255.

    Returns
    -------
    np.ndarray
        Clipped image with same shape and dtype as input.

    Raises
    ------
    TypeError
        If image is not a NumPy ndarray.
    ValueError
        If low >= high.
    """
    validate_image(image)
    if low >= high:
        raise ValueError(f"low ({low}) must be less than high ({high}).")
    return np.clip(image, low, high)


# ---------------------------------------------------------------------------
# Normalize
# ---------------------------------------------------------------------------

def normalize(image, mode="minmax"):
    """
    Normalize pixel values of an image.

    Parameters
    ----------
    image : np.ndarray
        Input image array (grayscale or RGB), any numeric dtype.
    mode : str
        Normalization mode. One of:
        - 'minmax'  : scales values to [0, 1].
        - 'zscore'  : zero mean, unit standard deviation.
        - 'uint8'   : scales values to [0, 255] as uint8.

    Returns
    -------
    np.ndarray
        Normalized image as float64 (or uint8 for mode='uint8').

    Raises
    ------
    TypeError
        If image is not a NumPy ndarray.
    ValueError
        If mode is not one of the supported options.

    Notes
    -----
    For 'minmax' and 'uint8', if max == min the output is all zeros.
    For 'zscore', if std == 0 the output is all zeros.
    """
    validate_image(image)
    image = image.astype(np.float64)

    if mode == "minmax":
        min_val = image.min()
        max_val = image.max()
        if max_val == min_val:
            return np.zeros_like(image)
        return (image - min_val) / (max_val - min_val)

    elif mode == "zscore":
        mean = image.mean()
        std = image.std()
        if std == 0:
            return np.zeros_like(image)
        return (image - mean) / std

    elif mode == "uint8":
        min_val = image.min()
        max_val = image.max()
        if max_val == min_val:
            return np.zeros_like(image, dtype=np.uint8)
        scaled = (image - min_val) / (max_val - min_val) * 255.0
        return scaled.astype(np.uint8)

    else:
        raise ValueError(f"mode must be 'minmax', 'zscore', or 'uint8', got '{mode}'.")


# ---------------------------------------------------------------------------
# Padding
# ---------------------------------------------------------------------------

def pad(image, pad_h, pad_w, mode="constant"):
    """
    Pad a 2D or 3D image array.

    Parameters
    ----------
    image : np.ndarray
        Input image (H, W) or (H, W, C).
    pad_h : int
        Number of pixels to pad on each side vertically (top and bottom).
    pad_w : int
        Number of pixels to pad on each side horizontally (left and right).
    mode : str
        Padding strategy. One of:
        - 'constant'  : pads with zeros.
        - 'reflect'   : pads by reflecting pixel values at the border.
        - 'replicate' : pads by repeating the edge pixel values.

    Returns
    -------
    np.ndarray
        Padded image with shape (H + 2*pad_h, W + 2*pad_w) or
        (H + 2*pad_h, W + 2*pad_w, C).

    Raises
    ------
    TypeError
        If image is not a NumPy ndarray.
    ValueError
        If pad_h or pad_w are negative, or mode is not supported.
    """
    validate_image(image)
    if pad_h < 0 or pad_w < 0:
        raise ValueError(f"pad_h and pad_w must be non-negative, got pad_h={pad_h}, pad_w={pad_w}.")

    if mode == "constant":
        np_mode = "constant"
    elif mode == "reflect":
        np_mode = "reflect"
    elif mode == "replicate":
        np_mode = "edge"
    else:
        raise ValueError(f"mode must be 'constant', 'reflect', or 'replicate', got '{mode}'.")

    if image.ndim == 2:
        return np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode=np_mode)
    else:
        return np.pad(image, ((pad_h, pad_h), (pad_w, pad_w), (0, 0)), mode=np_mode)


# ---------------------------------------------------------------------------
# 2D Convolution
# ---------------------------------------------------------------------------

def convolve2d(image, kernel, pad_mode="constant"):
    """
    Apply 2D convolution on a grayscale image using a given kernel.

    Parameters
    ----------
    image : np.ndarray
        2D grayscale image of shape (H, W).
    kernel : np.ndarray
        2D convolution kernel of shape (kH, kW) with odd dimensions.
    pad_mode : str
        Padding mode passed to pad(). One of 'constant', 'reflect', 'replicate'.

    Returns
    -------
    np.ndarray
        Convolved image of shape (H, W) as float64.

    Raises
    ------
    TypeError
        If image or kernel are not NumPy ndarrays.
    ValueError
        If image is not 2D, or kernel fails validation.

    Notes
    -----
    Boundary handling is done by padding the image before convolution
    so the output shape always matches the input shape.
    The kernel is flipped (true convolution, not cross-correlation).
    Vectorized using NumPy stride tricks for performance.
    """
    validate_image(image)
    validate_kernel(kernel)
    if image.ndim != 2:
        raise ValueError(f"convolve2d expects a 2D grayscale image, got shape {image.shape}.")

    image = image.astype(np.float64)
    kernel = kernel.astype(np.float64)

    kH, kW = kernel.shape
    pad_h = kH // 2
    pad_w = kW // 2

    padded = pad(image, pad_h, pad_w, mode=pad_mode)
    flipped_kernel = np.flip(kernel)

    H, W = image.shape
    output = np.zeros((H, W), dtype=np.float64)

    # Use stride tricks to create a sliding window view — avoids pixel loops
    shape = (H, W, kH, kW)
    strides = (
        padded.strides[0],
        padded.strides[1],
        padded.strides[0],
        padded.strides[1],
    )
    windows = np.lib.stride_tricks.as_strided(padded, shape=shape, strides=strides)
    output = np.tensordot(windows, flipped_kernel, axes=([2, 3], [0, 1]))

    return output