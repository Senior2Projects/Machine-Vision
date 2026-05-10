import numpy as np

from .utils import validate_image, clip_pixels
from .color import rgb_to_gray


# ---------------------------------------------------------------------------
# Bit-Plane Slicing
# ---------------------------------------------------------------------------

def bit_plane_slice(image, plane):
    """
    Extract a single bit plane from a grayscale image.

    Parameters
    ----------
    image : np.ndarray
        Grayscale image of shape (H, W), dtype uint8.
        If RGB is passed it will be converted to grayscale automatically.
    plane : int
        Bit plane to extract. Must be in range [0, 7].
        Plane 7 is the most significant bit (MSB).
        Plane 0 is the least significant bit (LSB).

    Returns
    -------
    np.ndarray
        Binary image of shape (H, W), dtype uint8, with values 0 or 255.
        Pixels where the selected bit is 1 become 255, others become 0.

    Raises
    ------
    TypeError
        If image is not a NumPy ndarray or plane is not an int.
    ValueError
        If plane is not in [0, 7].

    Notes
    -----
    Each pixel value is an 8-bit integer. Bit-plane slicing isolates
    one bit across all pixels using a bitwise AND with a mask:
        mask = 2^plane
        bit_plane = (image & mask) >> plane  -> 0 or 1
    Higher planes (6, 7) carry most of the visual information.
    Lower planes (0, 1, 2) carry fine detail and noise.
    """
    validate_image(image)
    if not isinstance(plane, int):
        raise TypeError(f"plane must be an int, got {type(plane).__name__}.")
    if not (0 <= plane <= 7):
        raise ValueError(f"plane must be in [0, 7], got {plane}.")

    if image.ndim == 3:
        image = rgb_to_gray(image)

    image = image.astype(np.uint8)
    mask = np.uint8(1 << plane)
    bit = (image & mask) >> plane
    return (bit * 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Histogram
# ---------------------------------------------------------------------------

def histogram(image, bins=256):
    """
    Compute the intensity histogram of a grayscale image.

    Parameters
    ----------
    image : np.ndarray
        Grayscale image of shape (H, W). If RGB is passed it will be
        converted to grayscale automatically.
    bins : int
        Number of histogram bins. Must be a positive integer.
        Default is 256 (one bin per intensity level).

    Returns
    -------
    counts : np.ndarray
        Array of shape (bins,) with the pixel count per bin.
    bin_edges : np.ndarray
        Array of shape (bins + 1,) with the bin edge values.

    Raises
    ------
    TypeError
        If image is not a NumPy ndarray or bins is not an int.
    ValueError
        If bins is not a positive integer.

    Notes
    -----
    The histogram spans the range [0, 255] regardless of the actual
    pixel value range in the image. This ensures consistent bin widths
    when comparing histograms across different images.
    """
    validate_image(image)
    if not isinstance(bins, int):
        raise TypeError(f"bins must be an int, got {type(bins).__name__}.")
    if bins <= 0:
        raise ValueError(f"bins must be a positive integer, got {bins}.")

    if image.ndim == 3:
        image = rgb_to_gray(image)

    image = image.astype(np.float64)
    counts, bin_edges = np.histogram(image.ravel(), bins=bins, range=(0, 255))
    return counts, bin_edges


# ---------------------------------------------------------------------------
# Histogram Equalization
# ---------------------------------------------------------------------------

def histogram_equalization(image):
    """
    Enhance image contrast using histogram equalization.

    Parameters
    ----------
    image : np.ndarray
        Grayscale image of shape (H, W), dtype uint8 expected.
        If RGB is passed it will be converted to grayscale automatically.

    Returns
    -------
    np.ndarray
        Contrast-enhanced grayscale image of shape (H, W), dtype uint8,
        with pixel values in [0, 255].

    Raises
    ------
    TypeError
        If image is not a NumPy ndarray.
    ValueError
        If image is not 2D or 3D.

    Notes
    -----
    Histogram equalization redistributes pixel intensities so that the
    output histogram is approximately uniform. The transform is:
        s = round( (L-1) * CDF(r) )
    where:
        L   = number of intensity levels (256)
        CDF = cumulative distribution function of the input histogram
        r   = input intensity level
        s   = output intensity level
    The CDF is normalized by the total number of pixels so it spans [0, 1].
    This is fully vectorized using a lookup table approach — no pixel loops.
    """
    validate_image(image)

    if image.ndim == 3:
        image = rgb_to_gray(image)

    image = image.astype(np.uint8)
    total_pixels = image.size

    # Compute histogram and CDF
    counts, _ = np.histogram(image.ravel(), bins=256, range=(0, 255))
    cdf = np.cumsum(counts) / total_pixels

    # Build lookup table: map each intensity [0-255] to equalized value
    lut = np.round(cdf * 255).astype(np.uint8)

    # Apply lookup table — fully vectorized
    equalized = lut[image]
    return equalized


# ---------------------------------------------------------------------------
# Contrast Stretching  (extra technique 1)
# ---------------------------------------------------------------------------

def contrast_stretching(image, in_low=None, in_high=None, out_low=0, out_high=255):
    """
    Stretch the contrast of an image by linearly rescaling intensity values.

    Parameters
    ----------
    image : np.ndarray
        Grayscale image of shape (H, W). If RGB is passed it will be
        converted to grayscale automatically.
    in_low : float or None
        Lower bound of the input intensity range. If None, uses image min.
    in_high : float or None
        Upper bound of the input intensity range. If None, uses image max.
    out_low : float
        Lower bound of the output intensity range. Default is 0.
    out_high : float
        Upper bound of the output intensity range. Default is 255.

    Returns
    -------
    np.ndarray
        Contrast-stretched image of shape (H, W), dtype uint8.

    Raises
    ------
    TypeError
        If image is not a NumPy ndarray or bounds are not numbers.
    ValueError
        If in_low >= in_high or out_low >= out_high.

    Notes
    -----
    The linear stretch formula maps each pixel r to output s:
        s = (r - in_low) / (in_high - in_low) * (out_high - out_low) + out_low
    Pixels outside [in_low, in_high] are clipped to [out_low, out_high].
    Unlike histogram equalization, this preserves the shape of the histogram
    while expanding its range — useful when the dynamic range is narrow.
    """
    validate_image(image)

    if image.ndim == 3:
        image = rgb_to_gray(image)

    image = image.astype(np.float64)

    if in_low is None:
        in_low = image.min()
    if in_high is None:
        in_high = image.max()

    if not isinstance(in_low, (int, float)):
        raise TypeError(f"in_low must be a number, got {type(in_low).__name__}.")
    if not isinstance(in_high, (int, float)):
        raise TypeError(f"in_high must be a number, got {type(in_high).__name__}.")
    if in_low >= in_high:
        raise ValueError(f"in_low ({in_low}) must be less than in_high ({in_high}).")
    if out_low >= out_high:
        raise ValueError(f"out_low ({out_low}) must be less than out_high ({out_high}).")

    stretched = (image - in_low) / (in_high - in_low) * (out_high - out_low) + out_low
    stretched = np.clip(stretched, out_low, out_high).astype(np.uint8)
    return stretched


# ---------------------------------------------------------------------------
# Laplacian Sharpening  (extra technique 2)
# ---------------------------------------------------------------------------

def laplacian_sharpening(image, strength=1.0):
    """
    Sharpen an image by adding a scaled Laplacian to the original.

    Parameters
    ----------
    image : np.ndarray
        Grayscale image of shape (H, W). If RGB is passed, sharpening
        is applied per channel.
    strength : float
        Scaling factor for the Laplacian. Must be > 0.
        Higher values produce stronger sharpening.
        Typical range: 0.5 to 2.0.

    Returns
    -------
    np.ndarray
        Sharpened image of same shape as input, dtype uint8,
        with pixel values clipped to [0, 255].

    Raises
    ------
    TypeError
        If image is not a NumPy ndarray or strength is not a number.
    ValueError
        If strength <= 0.

    Notes
    -----
    The Laplacian kernel used:
        K = [[ 0, -1,  0],
             [-1,  4, -1],
             [ 0, -1,  0]]
    The sharpened output is:
        output = original + strength * Laplacian(original)
    The Laplacian highlights regions of rapid intensity change (edges).
    Adding it back to the original amplifies those edges, producing
    a sharpened result. Output is clipped to valid uint8 range.
    """
    validate_image(image)
    if not isinstance(strength, (int, float)):
        raise TypeError(f"strength must be a number, got {type(strength).__name__}.")
    if strength <= 0:
        raise ValueError(f"strength must be > 0, got {strength}.")

    from .utils import convolve2d

    laplacian_kernel = np.array([[ 0, -1,  0],
                                 [-1,  4, -1],
                                 [ 0, -1,  0]], dtype=np.float64)

    def _sharpen_channel(channel):
        channel = channel.astype(np.float64)
        lap = convolve2d(channel, laplacian_kernel, pad_mode="reflect")
        sharpened = channel + strength * lap
        return np.clip(sharpened, 0, 255).astype(np.uint8)

    if image.ndim == 2:
        return _sharpen_channel(image)

    channels = [_sharpen_channel(image[:, :, c]) for c in range(image.shape[2])]
    return np.stack(channels, axis=-1)