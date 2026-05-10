import numpy as np

from .utils import validate_image, pad
from .color import rgb_to_gray


# ---------------------------------------------------------------------------
# Global Threshold
# ---------------------------------------------------------------------------

def global_threshold(image, threshold):
    """
    Apply a fixed global threshold to a grayscale image.

    Parameters
    ----------
    image : np.ndarray
        Grayscale image of shape (H, W). If RGB is passed it will be
        converted to grayscale automatically.
    threshold : float
        Threshold value in range [0, 255]. Pixels above this value
        become 255, pixels below become 0.

    Returns
    -------
    np.ndarray
        Binary image of shape (H, W), dtype uint8, with values 0 or 255.

    Raises
    ------
    TypeError
        If image is not a NumPy ndarray or threshold is not a number.
    ValueError
        If threshold is not in range [0, 255].

    Notes
    -----
    This is the simplest thresholding method. It works well when the
    image has a clear bimodal histogram (distinct foreground/background).
    For images with uneven lighting, use adaptive_threshold instead.
    """
    validate_image(image)
    if not isinstance(threshold, (int, float)):
        raise TypeError(f"threshold must be a number, got {type(threshold).__name__}.")
    if not (0 <= threshold <= 255):
        raise ValueError(f"threshold must be in [0, 255], got {threshold}.")

    if image.ndim == 3:
        image = rgb_to_gray(image)

    image = image.astype(np.float64)
    binary = np.where(image >= threshold, 255, 0).astype(np.uint8)
    return binary


# ---------------------------------------------------------------------------
# Otsu Threshold
# ---------------------------------------------------------------------------

def otsu_threshold(image):
    """
    Automatically determine the optimal threshold using Otsu's method.

    Parameters
    ----------
    image : np.ndarray
        Grayscale image of shape (H, W). If RGB is passed it will be
        converted to grayscale automatically.

    Returns
    -------
    binary : np.ndarray
        Binary image of shape (H, W), dtype uint8, with values 0 or 255.
    threshold : float
        The optimal threshold value found by Otsu's method.

    Raises
    ------
    TypeError
        If image is not a NumPy ndarray.
    ValueError
        If image is not 2D or 3D.

    Notes
    -----
    Otsu's method finds the threshold that minimizes intra-class variance,
    equivalently maximizing inter-class variance between foreground and
    background. The formula for inter-class variance is:
        sigma_b^2(t) = w0(t) * w1(t) * (mu0(t) - mu1(t))^2
    where w0, w1 are class probabilities and mu0, mu1 are class means.
    The threshold is swept over all 256 intensity levels and the one
    maximizing sigma_b^2 is selected. Fully vectorized — no loop over pixels.
    """
    validate_image(image)

    if image.ndim == 3:
        image = rgb_to_gray(image)

    image = image.astype(np.float64)
    total_pixels = image.size

    # Compute normalized histogram
    counts, _ = np.histogram(image.ravel(), bins=256, range=(0, 255))
    prob = counts / total_pixels

    # Sweep all possible thresholds vectorized
    levels = np.arange(256)
    w0 = np.cumsum(prob)                        # weight of background class
    w1 = 1.0 - w0                               # weight of foreground class

    mu0 = np.cumsum(prob * levels) / np.where(w0 == 0, 1, w0)
    mu_total = np.sum(prob * levels)
    mu1 = np.where(w1 == 0, 0, (mu_total - np.cumsum(prob * levels)) / np.where(w1 == 0, 1, w1))

    sigma_b_squared = w0 * w1 * (mu0 - mu1) ** 2

    optimal_threshold = float(np.argmax(sigma_b_squared))
    binary = np.where(image >= optimal_threshold, 255, 0).astype(np.uint8)

    return binary, optimal_threshold


# ---------------------------------------------------------------------------
# Adaptive Threshold
# ---------------------------------------------------------------------------

def adaptive_threshold(image, block_size=11, C=2, method="mean"):
    """
    Apply adaptive thresholding using local neighborhood statistics.

    Parameters
    ----------
    image : np.ndarray
        Grayscale image of shape (H, W). If RGB is passed it will be
        converted to grayscale automatically.
    block_size : int
        Size of the local neighborhood window. Must be a positive odd integer.
        Larger values capture broader lighting variations.
    C : float
        Constant subtracted from the local mean or gaussian-weighted mean.
        Positive values make thresholding more strict (fewer white pixels).
    method : str
        Local statistic to use as threshold. One of:
        - 'mean'     : threshold = mean of neighborhood - C.
        - 'gaussian' : threshold = gaussian-weighted mean of neighborhood - C.

    Returns
    -------
    np.ndarray
        Binary image of shape (H, W), dtype uint8, with values 0 or 255.

    Raises
    ------
    TypeError
        If image is not a NumPy ndarray, block_size is not an int,
        or C is not a number.
    ValueError
        If block_size is not a positive odd integer, or method is unsupported.

    Notes
    -----
    Unlike global thresholding, adaptive thresholding computes a different
    threshold for each pixel based on its local neighborhood. This handles
    images with non-uniform illumination much better.
    The gaussian method uses a gaussian-weighted neighborhood mean, giving
    more weight to pixels closer to the center — this reduces haloing effects
    compared to the plain mean method.
    """
    validate_image(image)
    if not isinstance(block_size, int):
        raise TypeError(f"block_size must be an int, got {type(block_size).__name__}.")
    if block_size <= 0 or block_size % 2 == 0:
        raise ValueError(f"block_size must be a positive odd integer, got {block_size}.")
    if not isinstance(C, (int, float)):
        raise TypeError(f"C must be a number, got {type(C).__name__}.")
    if method not in ("mean", "gaussian"):
        raise ValueError(f"method must be 'mean' or 'gaussian', got '{method}'.")

    if image.ndim == 3:
        image = rgb_to_gray(image)

    image = image.astype(np.float64)
    H, W = image.shape
    pad_size = block_size // 2

    if method == "mean":
        kernel = np.ones((block_size, block_size), dtype=np.float64) / (block_size ** 2)
    else:
        # Gaussian weights
        from .filtering import gaussian_kernel
        kernel = gaussian_kernel(block_size, sigma=block_size / 6.0)

    # Compute local threshold map using convolution
    from .utils import convolve2d
    local_mean = convolve2d(image, kernel, pad_mode="reflect")
    threshold_map = local_mean - C

    binary = np.where(image >= threshold_map, 255, 0).astype(np.uint8)
    return binary