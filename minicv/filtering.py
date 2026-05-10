import numpy as np

from .utils import validate_image, validate_kernel, convolve2d, pad


# ---------------------------------------------------------------------------
# Spatial Filter (main dispatcher)
# ---------------------------------------------------------------------------

def spatial_filter(image, kernel, pad_mode="constant"):
    """
    Apply a 2D convolution-based filter to a grayscale or RGB image.

    Parameters
    ----------
    image : np.ndarray
        Input image of shape (H, W) or (H, W, 3).
    kernel : np.ndarray
        2D convolution kernel of shape (kH, kW) with odd dimensions.
    pad_mode : str
        Padding mode. One of 'constant', 'reflect', 'replicate'.

    Returns
    -------
    np.ndarray
        Filtered image of same shape as input, dtype float64.

    Raises
    ------
    TypeError
        If image or kernel are not NumPy ndarrays.
    ValueError
        If image is not 2D or 3D, or kernel fails validation.

    Notes
    -----
    For RGB images, convolution is applied independently per channel.
    Output is not clipped — use clip_pixels() if uint8 output is needed.
    """
    validate_image(image)
    validate_kernel(kernel)

    if image.ndim == 2:
        return convolve2d(image, kernel, pad_mode=pad_mode)

    elif image.ndim == 3:
        channels = [
            convolve2d(image[:, :, c], kernel, pad_mode=pad_mode)
            for c in range(image.shape[2])
        ]
        return np.stack(channels, axis=-1)

    else:
        raise ValueError(f"image must be 2D or 3D, got shape {image.shape}.")


# ---------------------------------------------------------------------------
# Mean / Box Filter
# ---------------------------------------------------------------------------

def mean_filter(image, kernel_size=3, pad_mode="constant"):
    """
    Apply a mean (box) filter to smooth an image.

    Parameters
    ----------
    image : np.ndarray
        Input image of shape (H, W) or (H, W, 3).
    kernel_size : int
        Size of the square kernel. Must be a positive odd integer.
    pad_mode : str
        Padding mode. One of 'constant', 'reflect', 'replicate'.

    Returns
    -------
    np.ndarray
        Smoothed image of same shape as input, dtype float64.

    Raises
    ------
    TypeError
        If kernel_size is not an integer.
    ValueError
        If kernel_size is not positive or not odd.

    Notes
    -----
    The box kernel is a (k x k) matrix where every element = 1 / (k * k).
    Each output pixel is the arithmetic mean of its neighborhood.
    """
    if not isinstance(kernel_size, int):
        raise TypeError(f"kernel_size must be an int, got {type(kernel_size).__name__}.")
    if kernel_size <= 0 or kernel_size % 2 == 0:
        raise ValueError(f"kernel_size must be a positive odd integer, got {kernel_size}.")

    kernel = np.ones((kernel_size, kernel_size), dtype=np.float64) / (kernel_size ** 2)
    return spatial_filter(image, kernel, pad_mode=pad_mode)


# ---------------------------------------------------------------------------
# Gaussian Kernel
# ---------------------------------------------------------------------------

def gaussian_kernel(size, sigma):
    """
    Generate a 2D Gaussian kernel.

    Parameters
    ----------
    size : int
        Width and height of the kernel. Must be a positive odd integer.
    sigma : float
        Standard deviation of the Gaussian distribution. Must be > 0.

    Returns
    -------
    np.ndarray
        Normalized 2D Gaussian kernel of shape (size, size), dtype float64.
        Values sum to 1.0.

    Raises
    ------
    TypeError
        If size is not an int or sigma is not a number.
    ValueError
        If size is not a positive odd integer, or sigma <= 0.

    Notes
    -----
    The Gaussian formula used:
        G(x, y) = exp(-(x^2 + y^2) / (2 * sigma^2))
    The kernel is normalized by dividing by its sum so it acts as
    a weighted average (no brightness change on uniform regions).
    """
    if not isinstance(size, int):
        raise TypeError(f"size must be an int, got {type(size).__name__}.")
    if size <= 0 or size % 2 == 0:
        raise ValueError(f"size must be a positive odd integer, got {size}.")
    if not isinstance(sigma, (int, float)):
        raise TypeError(f"sigma must be a number, got {type(sigma).__name__}.")
    if sigma <= 0:
        raise ValueError(f"sigma must be > 0, got {sigma}.")

    half = size // 2
    ax = np.arange(-half, half + 1, dtype=np.float64)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx ** 2 + yy ** 2) / (2.0 * sigma ** 2))
    return kernel / kernel.sum()


# ---------------------------------------------------------------------------
# Gaussian Filter
# ---------------------------------------------------------------------------

def gaussian_filter(image, kernel_size=5, sigma=1.0, pad_mode="constant"):
    """
    Apply a Gaussian blur filter to an image.

    Parameters
    ----------
    image : np.ndarray
        Input image of shape (H, W) or (H, W, 3).
    kernel_size : int
        Size of the square Gaussian kernel. Must be a positive odd integer.
    sigma : float
        Standard deviation of the Gaussian. Must be > 0.
    pad_mode : str
        Padding mode. One of 'constant', 'reflect', 'replicate'.

    Returns
    -------
    np.ndarray
        Blurred image of same shape as input, dtype float64.

    Raises
    ------
    TypeError
        If kernel_size is not an int or sigma is not a number.
    ValueError
        If kernel_size or sigma are invalid.

    Notes
    -----
    Larger sigma = more blur. kernel_size should be at least 6*sigma
    to capture the full distribution (rule of thumb: size = 6*sigma + 1).
    """
    kernel = gaussian_kernel(kernel_size, sigma)
    return spatial_filter(image, kernel, pad_mode=pad_mode)


# ---------------------------------------------------------------------------
# Median Filter
# ---------------------------------------------------------------------------

def median_filter(image, kernel_size=3, pad_mode="constant"):
    """
    Apply a median filter to remove noise from an image.

    Parameters
    ----------
    image : np.ndarray
        Input image of shape (H, W) or (H, W, 3).
    kernel_size : int
        Size of the square neighborhood window. Must be a positive odd integer.
    pad_mode : str
        Padding mode. One of 'constant', 'reflect', 'replicate'.

    Returns
    -------
    np.ndarray
        Filtered image of same shape as input, dtype float64.

    Raises
    ------
    TypeError
        If kernel_size is not an int.
    ValueError
        If kernel_size is not a positive odd integer.

    Notes
    -----
    Loop justification: the median operation is inherently non-linear and
    cannot be expressed as a dot product (unlike convolution). A loop over
    spatial positions is therefore unavoidable. The loop is kept efficient
    by using NumPy's stride_tricks to extract all windows at once and then
    computing the median across the window axis in a single vectorized call.
    No pixel-by-pixel Python loop is used.
    """
    if not isinstance(kernel_size, int):
        raise TypeError(f"kernel_size must be an int, got {type(kernel_size).__name__}.")
    if kernel_size <= 0 or kernel_size % 2 == 0:
        raise ValueError(f"kernel_size must be a positive odd integer, got {kernel_size}.")

    def _median_gray(channel):
        channel = channel.astype(np.float64)
        pad_h = pad_w = kernel_size // 2
        padded = pad(channel, pad_h, pad_w, mode=pad_mode)
        H, W = channel.shape

        # Build sliding window view — shape (H, W, kH, kW)
        shape = (H, W, kernel_size, kernel_size)
        strides = (
            padded.strides[0],
            padded.strides[1],
            padded.strides[0],
            padded.strides[1],
        )
        windows = np.lib.stride_tricks.as_strided(padded, shape=shape, strides=strides)

        # Compute median over the window axes — vectorized, no pixel loop
        return np.median(windows.reshape(H, W, -1), axis=-1)

    if image.ndim == 2:
        return _median_gray(image)

    channels = [_median_gray(image[:, :, c]) for c in range(image.shape[2])]
    return np.stack(channels, axis=-1)


# ---------------------------------------------------------------------------
# Sobel Gradients
# ---------------------------------------------------------------------------

def sobel_gradients(image):
    """
    Compute Sobel edge gradients of a grayscale image.

    Parameters
    ----------
    image : np.ndarray
        Grayscale image of shape (H, W). If RGB is passed it will be
        converted to grayscale automatically.

    Returns
    -------
    Gx : np.ndarray
        Horizontal gradient map, shape (H, W), dtype float64.
    Gy : np.ndarray
        Vertical gradient map, shape (H, W), dtype float64.
    magnitude : np.ndarray
        Gradient magnitude: sqrt(Gx^2 + Gy^2), shape (H, W), dtype float64.

    Raises
    ------
    TypeError
        If image is not a NumPy ndarray.
    ValueError
        If image is not 2D or 3D.

    Notes
    -----
    Sobel kernels:
        Kx = [[-1, 0, 1],        Ky = [[-1, -2, -1],
              [-2, 0, 2],               [ 0,  0,  0],
              [-1, 0, 1]]               [ 1,  2,  1]]
    Gx detects vertical edges, Gy detects horizontal edges.
    Magnitude combines both into a single edge strength map.
    """
    validate_image(image)

    if image.ndim == 3:
        from .color import rgb_to_gray
        image = rgb_to_gray(image)

    Kx = np.array([[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]], dtype=np.float64)

    Ky = np.array([[-1, -2, -1],
                   [ 0,  0,  0],
                   [ 1,  2,  1]], dtype=np.float64)

    Gx = convolve2d(image, Kx, pad_mode="replicate")
    Gy = convolve2d(image, Ky, pad_mode="replicate")
    magnitude = np.sqrt(Gx ** 2 + Gy ** 2)

    return Gx, Gy, magnitude