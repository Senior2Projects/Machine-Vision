import numpy as np

from .utils import validate_image
from .color import rgb_to_gray
from .filtering import sobel_gradients


# ---------------------------------------------------------------------------
# Global Descriptor 1: Color Histogram Descriptor
# ---------------------------------------------------------------------------

def color_histogram_descriptor(image, bins=32):
    """
    Compute a normalized color histogram descriptor for an image.

    Parameters
    ----------
    image : np.ndarray
        Input image of shape (H, W) or (H, W, 3).
    bins : int
        Number of histogram bins per channel. Must be a positive integer.
        Default is 32.

    Returns
    -------
    np.ndarray
        1D feature vector of shape:
        - (bins,)     for grayscale images.
        - (bins * 3,) for RGB images (R, G, B histograms concatenated).
        Values are normalized to sum to 1.0.

    Raises
    ------
    TypeError
        If image is not a NumPy ndarray or bins is not an int.
    ValueError
        If bins is not a positive integer.

    Notes
    -----
    For RGB images, a histogram is computed independently for each channel
    and the three histograms are concatenated into a single vector.
    Normalization ensures the descriptor is invariant to image size.
    This is a global descriptor — it captures the overall color distribution
    but discards spatial information.
    """
    validate_image(image)
    if not isinstance(bins, int):
        raise TypeError(f"bins must be an int, got {type(bins).__name__}.")
    if bins <= 0:
        raise ValueError(f"bins must be a positive integer, got {bins}.")

    image = image.astype(np.float64)

    if image.ndim == 2:
        counts, _ = np.histogram(image.ravel(), bins=bins, range=(0, 255))
        counts = counts.astype(np.float64)
        total = counts.sum()
        return counts / total if total > 0 else counts

    descriptors = []
    for c in range(image.shape[2]):
        counts, _ = np.histogram(image[:, :, c].ravel(), bins=bins, range=(0, 255))
        counts = counts.astype(np.float64)
        total = counts.sum()
        descriptors.append(counts / total if total > 0 else counts)

    return np.concatenate(descriptors)


# ---------------------------------------------------------------------------
# Global Descriptor 2: Hu Moments Descriptor
# ---------------------------------------------------------------------------

def hu_moments_descriptor(image):
    """
    Compute the seven Hu moment invariants of a grayscale image.

    Parameters
    ----------
    image : np.ndarray
        Input image of shape (H, W) or (H, W, 3).
        If RGB is passed it will be converted to grayscale automatically.

    Returns
    -------
    np.ndarray
        1D feature vector of shape (7,) containing the seven Hu moments.
        Values are log-transformed for numerical stability:
            hu_i = -sign(h_i) * log10(|h_i| + eps)

    Raises
    ------
    TypeError
        If image is not a NumPy ndarray.
    ValueError
        If image is not 2D or 3D.

    Notes
    -----
    Hu moments are derived from normalized central moments and are
    invariant to translation, scale, and rotation.

    Raw moments:
        M_pq = sum_x sum_y (x^p * y^q * I(x, y))

    Central moments (translation-invariant):
        mu_pq = sum_x sum_y ((x - x_bar)^p * (y - y_bar)^q * I(x, y))

    Normalized central moments (scale-invariant):
        eta_pq = mu_pq / mu_00^(1 + (p+q)/2)

    The seven Hu moments are specific combinations of eta values that
    are also rotation-invariant. See Hu (1962) for the full formulas.
    Log transform is applied to compress the dynamic range, which is
    standard practice when using Hu moments as ML features.
    """
    validate_image(image)

    if image.ndim == 3:
        image = rgb_to_gray(image)

    image = image.astype(np.float64)
    H, W = image.shape

    # Pixel coordinate grids
    x = np.arange(W, dtype=np.float64)
    y = np.arange(H, dtype=np.float64)
    xx, yy = np.meshgrid(x, y)

    # Raw moments
    def raw_moment(p, q):
        return np.sum((xx ** p) * (yy ** q) * image)

    m00 = raw_moment(0, 0)
    if m00 == 0:
        return np.zeros(7, dtype=np.float64)

    m10 = raw_moment(1, 0)
    m01 = raw_moment(0, 1)

    # Centroid
    x_bar = m10 / m00
    y_bar = m01 / m00

    # Centered coordinate grids
    xc = xx - x_bar
    yc = yy - y_bar

    # Central moments
    def central_moment(p, q):
        return np.sum((xc ** p) * (yc ** q) * image)

    mu00 = m00
    mu20 = central_moment(2, 0)
    mu02 = central_moment(0, 2)
    mu11 = central_moment(1, 1)
    mu30 = central_moment(3, 0)
    mu03 = central_moment(0, 3)
    mu21 = central_moment(2, 1)
    mu12 = central_moment(1, 2)

    # Normalized central moments
    def eta(p, q, mu_pq):
        return mu_pq / (mu00 ** (1 + (p + q) / 2.0))

    n20 = eta(2, 0, mu20)
    n02 = eta(0, 2, mu02)
    n11 = eta(1, 1, mu11)
    n30 = eta(3, 0, mu30)
    n03 = eta(0, 3, mu03)
    n21 = eta(2, 1, mu21)
    n12 = eta(1, 2, mu12)

    # Seven Hu moment invariants
    h = np.zeros(7, dtype=np.float64)
    h[0] = n20 + n02
    h[1] = (n20 - n02) ** 2 + 4 * n11 ** 2
    h[2] = (n30 - 3 * n12) ** 2 + (3 * n21 - n03) ** 2
    h[3] = (n30 + n12) ** 2 + (n21 + n03) ** 2
    h[4] = (n30 - 3 * n12) * (n30 + n12) * ((n30 + n12) ** 2 - 3 * (n21 + n03) ** 2) + \
           (3 * n21 - n03) * (n21 + n03) * (3 * (n30 + n12) ** 2 - (n21 + n03) ** 2)
    h[5] = (n20 - n02) * ((n30 + n12) ** 2 - (n21 + n03) ** 2) + \
           4 * n11 * (n30 + n12) * (n21 + n03)
    h[6] = (3 * n21 - n03) * (n30 + n12) * ((n30 + n12) ** 2 - 3 * (n21 + n03) ** 2) - \
           (n30 - 3 * n12) * (n21 + n03) * (3 * (n30 + n12) ** 2 - (n21 + n03) ** 2)

    # Log transform for numerical stability
    eps = 1e-10
    hu = -np.sign(h) * np.log10(np.abs(h) + eps)
    return hu


# ---------------------------------------------------------------------------
# Gradient Descriptor 1: HOG (Histogram of Oriented Gradients)
# ---------------------------------------------------------------------------

def hog_descriptor(image, cell_size=8, bins=9):
    """
    Compute a Histogram of Oriented Gradients (HOG) descriptor.

    Parameters
    ----------
    image : np.ndarray
        Input image of shape (H, W) or (H, W, 3).
        If RGB is passed it will be converted to grayscale automatically.
    cell_size : int
        Size of each cell in pixels (cell_size x cell_size).
        Must be a positive integer. Default is 8.
    bins : int
        Number of orientation bins in [0, 180) degrees.
        Must be a positive integer. Default is 9.

    Returns
    -------
    np.ndarray
        1D HOG feature vector of shape (n_cells_y * n_cells_x * bins,).
        Values are L2-normalized per cell.

    Raises
    ------
    TypeError
        If image is not a NumPy ndarray, or cell_size/bins are not ints.
    ValueError
        If cell_size or bins are not positive integers.

    Notes
    -----
    HOG steps:
        1. Convert to grayscale and compute Sobel gradients (Gx, Gy).
        2. Compute magnitude and orientation:
               magnitude = sqrt(Gx^2 + Gy^2)
               orientation = arctan2(|Gy|, Gx) in degrees [0, 180)
        3. Divide image into non-overlapping cells of size cell_size x cell_size.
        4. For each cell, build a weighted histogram of orientations
           where each gradient vote is weighted by its magnitude.
        5. L2-normalize each cell histogram.
        6. Concatenate all cell histograms into one feature vector.

    Unsigned orientations [0, 180) are used (not [0, 360)) as they are
    more robust for object recognition tasks.
    """
    validate_image(image)
    if not isinstance(cell_size, int):
        raise TypeError(f"cell_size must be an int, got {type(cell_size).__name__}.")
    if cell_size <= 0:
        raise ValueError(f"cell_size must be a positive integer, got {cell_size}.")
    if not isinstance(bins, int):
        raise TypeError(f"bins must be an int, got {type(bins).__name__}.")
    if bins <= 0:
        raise ValueError(f"bins must be a positive integer, got {bins}.")

    if image.ndim == 3:
        image = rgb_to_gray(image)

    image = image.astype(np.float64)
    H, W = image.shape

    # Crop to be divisible by cell_size
    H_crop = (H // cell_size) * cell_size
    W_crop = (W // cell_size) * cell_size
    image = image[:H_crop, :W_crop]

    Gx, Gy, magnitude = sobel_gradients(image)

    # Unsigned orientation in degrees [0, 180)
    orientation = np.rad2deg(np.arctan2(np.abs(Gy), Gx)) % 180

    n_cells_y = H_crop // cell_size
    n_cells_x = W_crop // cell_size
    bin_width = 180.0 / bins

    hog = np.zeros((n_cells_y, n_cells_x, bins), dtype=np.float64)

    # Assign each pixel's gradient vote to the correct bin — vectorized
    bin_indices = (orientation / bin_width).astype(np.int64) % bins

    for by in range(n_cells_y):
        for bx in range(n_cells_x):
            r0, r1 = by * cell_size, (by + 1) * cell_size
            c0, c1 = bx * cell_size, (bx + 1) * cell_size
            cell_mag = magnitude[r0:r1, c0:c1].ravel()
            cell_bin = bin_indices[r0:r1, c0:c1].ravel()
            hog[by, bx] = np.bincount(cell_bin, weights=cell_mag, minlength=bins)

    # L2-normalize each cell
    norms = np.linalg.norm(hog, axis=-1, keepdims=True)
    hog = hog / (norms + 1e-10)

    return hog.ravel()


# ---------------------------------------------------------------------------
# Gradient Descriptor 2: LBP (Local Binary Pattern)
# ---------------------------------------------------------------------------

def lbp_descriptor(image, radius=1, n_points=8, bins=256):
    """
    Compute a Local Binary Pattern (LBP) histogram descriptor.

    Parameters
    ----------
    image : np.ndarray
        Input image of shape (H, W) or (H, W, 3).
        If RGB is passed it will be converted to grayscale automatically.
    radius : int
        Radius of the circular LBP neighborhood. Must be a positive integer.
        Default is 1.
    n_points : int
        Number of sampling points on the circle of given radius.
        Must be a positive integer. Default is 8.
    bins : int
        Number of histogram bins for the LBP code histogram.
        Default is 256 (covers all 8-bit LBP codes when n_points=8).

    Returns
    -------
    np.ndarray
        1D normalized LBP histogram of shape (bins,).
        Values sum to 1.0.

    Raises
    ------
    TypeError
        If image is not a NumPy ndarray, or radius/n_points/bins are not ints.
    ValueError
        If radius, n_points, or bins are not positive integers.

    Notes
    -----
    LBP algorithm:
        1. For each pixel, sample n_points neighbors on a circle of
           the given radius at equal angular intervals.
        2. Compare each neighbor intensity to the center pixel.
        3. Encode comparisons as a binary string (1 if neighbor >= center,
           0 otherwise), read clockwise.
        4. Convert the binary string to a decimal LBP code.
        5. Build a histogram of all LBP codes across the image.

    Neighbor coordinates are computed analytically:
        x_i = x + radius * cos(2 * pi * i / n_points)
        y_i = y - radius * sin(2 * pi * i / n_points)

    Bilinear interpolation is used to sample neighbor values at
    non-integer coordinates, making the descriptor more accurate.

    Loop justification: the outer loop iterates over n_points neighbors
    (default 8 iterations), not over pixels. Each iteration is fully
    vectorized across all pixels using NumPy. This is the standard
    efficient implementation of LBP.
    """
    validate_image(image)
    if not isinstance(radius, int):
        raise TypeError(f"radius must be an int, got {type(radius).__name__}.")
    if radius <= 0:
        raise ValueError(f"radius must be a positive integer, got {radius}.")
    if not isinstance(n_points, int):
        raise TypeError(f"n_points must be an int, got {type(n_points).__name__}.")
    if n_points <= 0:
        raise ValueError(f"n_points must be a positive integer, got {n_points}.")
    if not isinstance(bins, int):
        raise TypeError(f"bins must be an int, got {type(bins).__name__}.")
    if bins <= 0:
        raise ValueError(f"bins must be a positive integer, got {bins}.")

    if image.ndim == 3:
        image = rgb_to_gray(image)

    image = image.astype(np.float64)
    H, W = image.shape

    lbp_map = np.zeros((H, W), dtype=np.float64)

    # Row and column coordinate grids
    rows = np.arange(H, dtype=np.float64)
    cols = np.arange(W, dtype=np.float64)
    col_grid, row_grid = np.meshgrid(cols, rows)

    # Loop over n_points neighbors only (not over pixels)
    for i in range(n_points):
        angle = 2 * np.pi * i / n_points
        n_col = col_grid + radius * np.cos(angle)   # neighbor col coords
        n_row = row_grid - radius * np.sin(angle)   # neighbor row coords

        # Bilinear interpolation of neighbor values — vectorized
        n_col_c = np.clip(n_col, 0, W - 1)
        n_row_c = np.clip(n_row, 0, H - 1)

        c0 = np.clip(np.floor(n_col_c).astype(np.int64), 0, W - 1)
        c1 = np.clip(c0 + 1,                              0, W - 1)
        r0 = np.clip(np.floor(n_row_c).astype(np.int64), 0, H - 1)
        r1 = np.clip(r0 + 1,                              0, H - 1)

        dc = n_col_c - c0
        dr = n_row_c - r0

        neighbor_val = (
            (1 - dr) * (1 - dc) * image[r0, c0] +
            (1 - dr) *      dc  * image[r0, c1] +
                 dr  * (1 - dc) * image[r1, c0] +
                 dr  *      dc  * image[r1, c1]
        )

        # Add bit contribution if neighbor >= center
        bit = (neighbor_val >= image).astype(np.float64)
        lbp_map += bit * (2 ** i)

    # Build normalized histogram of LBP codes
    lbp_int = lbp_map.astype(np.int64).ravel()
    counts = np.bincount(lbp_int, minlength=bins)[:bins].astype(np.float64)
    total = counts.sum()
    return counts / total if total > 0 else counts