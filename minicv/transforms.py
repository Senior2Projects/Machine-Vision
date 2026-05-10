import numpy as np

from .utils import validate_image


# ---------------------------------------------------------------------------
# Resize
# ---------------------------------------------------------------------------

def resize(image, out_h, out_w, method="bilinear"):
    """
    Resize an image to a target height and width.

    Parameters
    ----------
    image : np.ndarray
        Input image of shape (H, W) or (H, W, 3).
    out_h : int
        Target output height in pixels. Must be a positive integer.
    out_w : int
        Target output width in pixels. Must be a positive integer.
    method : str
        Interpolation method. One of:
        - 'nearest'  : nearest-neighbor interpolation (fast, blocky).
        - 'bilinear' : bilinear interpolation (smoother, default).

    Returns
    -------
    np.ndarray
        Resized image of shape (out_h, out_w) or (out_h, out_w, 3),
        dtype float64.

    Raises
    ------
    TypeError
        If image is not a NumPy ndarray, or out_h/out_w are not ints.
    ValueError
        If out_h or out_w are not positive, or method is unsupported.

    Notes
    -----
    Nearest-neighbor maps each output pixel to the closest input pixel:
        src_x = round(x * (W - 1) / (out_w - 1))
        src_y = round(y * (H - 1) / (out_h - 1))

    Bilinear interpolation maps each output pixel to a weighted average
    of the four surrounding input pixels:
        dx = src_x - floor(src_x)
        dy = src_y - floor(src_y)
        out = (1-dy)(1-dx)*I(y0,x0) + (1-dy)*dx*I(y0,x1)
            +    dy*(1-dx)*I(y1,x0) +    dy *dx*I(y1,x1)

    Both methods are fully vectorized using NumPy meshgrids — no pixel loops.
    """
    validate_image(image)
    if not isinstance(out_h, int) or not isinstance(out_w, int):
        raise TypeError(f"out_h and out_w must be ints, got {type(out_h).__name__} and {type(out_w).__name__}.")
    if out_h <= 0 or out_w <= 0:
        raise ValueError(f"out_h and out_w must be positive, got out_h={out_h}, out_w={out_w}.")
    if method not in ("nearest", "bilinear"):
        raise ValueError(f"method must be 'nearest' or 'bilinear', got '{method}'.")

    image = image.astype(np.float64)
    H, W = image.shape[:2]

    # Output pixel grid
    out_rows = np.arange(out_h, dtype=np.float64)
    out_cols = np.arange(out_w, dtype=np.float64)

    # Map output coords to input coords
    src_y = out_rows * (H - 1) / (out_h - 1) if out_h > 1 else np.zeros(out_h)
    src_x = out_cols * (W - 1) / (out_w - 1) if out_w > 1 else np.zeros(out_w)

    src_yy, src_xx = np.meshgrid(src_y, src_x, indexing="ij")  # (out_h, out_w)

    if method == "nearest":
        ny = np.clip(np.round(src_yy).astype(np.int64), 0, H - 1)
        nx = np.clip(np.round(src_xx).astype(np.int64), 0, W - 1)

        if image.ndim == 2:
            return image[ny, nx]
        return image[ny, nx, :]

    else:  # bilinear
        y0 = np.clip(np.floor(src_yy).astype(np.int64), 0, H - 1)
        y1 = np.clip(y0 + 1,                             0, H - 1)
        x0 = np.clip(np.floor(src_xx).astype(np.int64), 0, W - 1)
        x1 = np.clip(x0 + 1,                             0, W - 1)

        dy = src_yy - y0
        dx = src_xx - x0

        if image.ndim == 2:
            return (
                (1 - dy) * (1 - dx) * image[y0, x0] +
                (1 - dy) *      dx  * image[y0, x1] +
                     dy  * (1 - dx) * image[y1, x0] +
                     dy  *      dx  * image[y1, x1]
            )

        # RGB: expand dy/dx for broadcasting over channels
        dy = dy[:, :, np.newaxis]
        dx = dx[:, :, np.newaxis]
        return (
            (1 - dy) * (1 - dx) * image[y0, x0, :] +
            (1 - dy) *      dx  * image[y0, x1, :] +
                 dy  * (1 - dx) * image[y1, x0, :] +
                 dy  *      dx  * image[y1, x1, :]
        )


# ---------------------------------------------------------------------------
# Rotate
# ---------------------------------------------------------------------------

def rotate(image, angle, method="bilinear"):
    """
    Rotate an image about its center by a given angle.

    Parameters
    ----------
    image : np.ndarray
        Input image of shape (H, W) or (H, W, 3).
    angle : float
        Rotation angle in degrees. Positive values rotate counter-clockwise.
    method : str
        Interpolation method for sampling. One of:
        - 'nearest'  : nearest-neighbor interpolation.
        - 'bilinear' : bilinear interpolation (default, smoother).

    Returns
    -------
    np.ndarray
        Rotated image of same shape as input, dtype float64.
        Pixels outside the original image boundary are filled with 0.

    Raises
    ------
    TypeError
        If image is not a NumPy ndarray or angle is not a number.
    ValueError
        If method is not 'nearest' or 'bilinear'.

    Notes
    -----
    Uses inverse mapping: for each output pixel (x', y'), compute the
    corresponding source pixel (x, y) in the original image using:
        x = cos(a)*(x' - cx) + sin(a)*(y' - cy) + cx
        y = -sin(a)*(x' - cx) + cos(a)*(y' - cy) + cy
    where (cx, cy) is the image center and a is the rotation angle.
    Inverse mapping avoids holes in the output that forward mapping
    would produce. Fully vectorized — no pixel loops.
    """
    validate_image(image)
    if not isinstance(angle, (int, float)):
        raise TypeError(f"angle must be a number, got {type(angle).__name__}.")
    if method not in ("nearest", "bilinear"):
        raise ValueError(f"method must be 'nearest' or 'bilinear', got '{method}'.")

    image = image.astype(np.float64)
    H, W = image.shape[:2]

    angle_rad = np.deg2rad(angle)
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)

    cx = (W - 1) / 2.0
    cy = (H - 1) / 2.0

    # Output pixel grid
    out_cols = np.arange(W, dtype=np.float64)
    out_rows = np.arange(H, dtype=np.float64)
    xx, yy = np.meshgrid(out_cols, out_rows)  # (H, W)

    # Inverse rotation mapping
    xc = xx - cx
    yc = yy - cy
    src_x = cos_a * xc + sin_a * yc + cx
    src_y = -sin_a * xc + cos_a * yc + cy

    # Mask pixels that fall outside the source image
    valid = (
        (src_x >= 0) & (src_x <= W - 1) &
        (src_y >= 0) & (src_y <= H - 1)
    )

    if method == "nearest":
        nx = np.clip(np.round(src_x).astype(np.int64), 0, W - 1)
        ny = np.clip(np.round(src_y).astype(np.int64), 0, H - 1)

        if image.ndim == 2:
            output = np.zeros((H, W), dtype=np.float64)
            output[valid] = image[ny[valid], nx[valid]]
        else:
            output = np.zeros((H, W, image.shape[2]), dtype=np.float64)
            output[valid] = image[ny[valid], nx[valid], :]
        return output

    else:  # bilinear
        x0 = np.clip(np.floor(src_x).astype(np.int64), 0, W - 1)
        x1 = np.clip(x0 + 1,                            0, W - 1)
        y0 = np.clip(np.floor(src_y).astype(np.int64), 0, H - 1)
        y1 = np.clip(y0 + 1,                            0, H - 1)

        dx = src_x - x0
        dy = src_y - y0

        if image.ndim == 2:
            output = np.zeros((H, W), dtype=np.float64)
            output[valid] = (
                (1 - dy[valid]) * (1 - dx[valid]) * image[y0[valid], x0[valid]] +
                (1 - dy[valid]) *      dx[valid]  * image[y0[valid], x1[valid]] +
                     dy[valid]  * (1 - dx[valid]) * image[y1[valid], x0[valid]] +
                     dy[valid]  *      dx[valid]  * image[y1[valid], x1[valid]]
            )
        else:
            output = np.zeros((H, W, image.shape[2]), dtype=np.float64)
            output[valid] = (
                (1 - dy[valid, np.newaxis]) * (1 - dx[valid, np.newaxis]) * image[y0[valid], x0[valid], :] +
                (1 - dy[valid, np.newaxis]) *      dx[valid, np.newaxis]  * image[y0[valid], x1[valid], :] +
                     dy[valid, np.newaxis]  * (1 - dx[valid, np.newaxis]) * image[y1[valid], x0[valid], :] +
                     dy[valid, np.newaxis]  *      dx[valid, np.newaxis]  * image[y1[valid], x1[valid], :]
            )
        return output


# ---------------------------------------------------------------------------
# Translation
# ---------------------------------------------------------------------------

def translate(image, shift_x, shift_y):
    """
    Translate an image by a given number of pixels in x and y directions.

    Parameters
    ----------
    image : np.ndarray
        Input image of shape (H, W) or (H, W, 3).
    shift_x : int
        Number of pixels to shift horizontally.
        Positive shifts right, negative shifts left.
    shift_y : int
        Number of pixels to shift vertically.
        Positive shifts down, negative shifts up.

    Returns
    -------
    np.ndarray
        Translated image of same shape as input, dtype float64.
        Areas uncovered by the shift are filled with 0.

    Raises
    ------
    TypeError
        If image is not a NumPy ndarray, or shift_x/shift_y are not ints.

    Notes
    -----
    Translation is implemented using NumPy array slicing — no loops,
    no interpolation needed since shifts are integer pixel offsets.
    The output canvas is the same size as the input. Content shifted
    outside the canvas boundary is discarded.
    """
    validate_image(image)
    if not isinstance(shift_x, int) or not isinstance(shift_y, int):
        raise TypeError(
            f"shift_x and shift_y must be ints, "
            f"got {type(shift_x).__name__} and {type(shift_y).__name__}."
        )

    image = image.astype(np.float64)
    H, W = image.shape[:2]

    if image.ndim == 2:
        output = np.zeros((H, W), dtype=np.float64)
    else:
        output = np.zeros((H, W, image.shape[2]), dtype=np.float64)

    # Compute source and destination slice ranges
    src_row_start  = max(0, -shift_y)
    src_row_end    = min(H, H - shift_y)
    src_col_start  = max(0, -shift_x)
    src_col_end    = min(W, W - shift_x)

    dst_row_start  = max(0, shift_y)
    dst_row_end    = dst_row_start + (src_row_end - src_row_start)
    dst_col_start  = max(0, shift_x)
    dst_col_end    = dst_col_start + (src_col_end - src_col_start)

    output[dst_row_start:dst_row_end, dst_col_start:dst_col_end] = \
        image[src_row_start:src_row_end, src_col_start:src_col_end]

    return output