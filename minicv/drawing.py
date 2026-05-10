import numpy as np

from .utils import validate_image


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _validate_color(color, image):
    """
    Validate and normalize a color value for the given image.

    Parameters
    ----------
    color : int or tuple
        For grayscale images: a single int in [0, 255].
        For RGB images: a tuple of 3 ints, each in [0, 255].
    image : np.ndarray
        The image the color will be drawn on.

    Returns
    -------
    int or tuple
        Validated color value.

    Raises
    ------
    TypeError
        If color type does not match image type.
    ValueError
        If color values are out of range.
    """
    if image.ndim == 2:
        if not isinstance(color, (int, float)):
            raise TypeError(
                f"Grayscale image expects a scalar color, got {type(color).__name__}."
            )
        if not (0 <= color <= 255):
            raise ValueError(f"color must be in [0, 255], got {color}.")
    else:
        if not isinstance(color, (tuple, list)) or len(color) != 3:
            raise TypeError(
                f"RGB image expects a tuple of 3 ints for color, got {color}."
            )
        for i, c in enumerate(color):
            if not (0 <= c <= 255):
                raise ValueError(f"color channel {i} must be in [0, 255], got {c}.")
    return color


def _validate_thickness(thickness):
    """
    Validate thickness parameter for drawing functions.

    Parameters
    ----------
    thickness : int
        Line or shape thickness. Must be a positive integer.

    Raises
    ------
    TypeError
        If thickness is not an int.
    ValueError
        If thickness is not positive.
    """
    if not isinstance(thickness, int):
        raise TypeError(f"thickness must be an int, got {type(thickness).__name__}.")
    if thickness <= 0:
        raise ValueError(f"thickness must be a positive integer, got {thickness}.")


def _fill_rect(image, r0, r1, c0, c1, color):
    """
    Fill a rectangle on the image safely, clipping to canvas boundaries.

    Parameters
    ----------
    image : np.ndarray
        Image to draw on.
    r0, r1 : int
        Row start and end (before clipping).
    c0, c1 : int
        Column start and end (before clipping).
    color : int or tuple
        Color value.

    Notes
    -----
    Clipping is done here in one place for all drawing functions.
    If the rectangle is fully outside the canvas, nothing is drawn.
    """
    H, W = image.shape[:2]
    r0c = max(0, r0)
    r1c = min(H, r1)
    c0c = max(0, c0)
    c1c = min(W, c1)
    if r0c >= r1c or c0c >= c1c:
        return
    image[r0c:r1c, c0c:c1c] = color


# ---------------------------------------------------------------------------
# Point
# ---------------------------------------------------------------------------

def draw_point(image, x, y, color, thickness=1):
    """
    Draw a point (filled square) on an image.

    Parameters
    ----------
    image : np.ndarray
        Image array of shape (H, W) or (H, W, 3). Modified in place.
    x : int
        Column coordinate of the point center.
    y : int
        Row coordinate of the point center.
    color : int or tuple
        Grayscale scalar or RGB tuple (R, G, B). Values in [0, 255].
    thickness : int
        Side length of the square drawn around the point. Default is 1.

    Returns
    -------
    np.ndarray
        The image with the point drawn (same array, modified in place).

    Raises
    ------
    TypeError
        If image is not a NumPy ndarray, x/y are not ints,
        or color/thickness types are invalid.
    ValueError
        If color values are out of range or thickness is not positive.

    Notes
    -----
    Points outside the canvas are silently clipped — no error is raised.
    thickness=1 draws a single pixel. thickness=3 draws a 3x3 square.
    """
    validate_image(image)
    _validate_color(color, image)
    _validate_thickness(thickness)
    if not isinstance(x, int) or not isinstance(y, int):
        raise TypeError(f"x and y must be ints, got {type(x).__name__} and {type(y).__name__}.")

    half = thickness // 2
    _fill_rect(image, y - half, y - half + thickness, x - half, x - half + thickness, color)
    return image


# ---------------------------------------------------------------------------
# Line (Bresenham)
# ---------------------------------------------------------------------------

def draw_line(image, x0, y0, x1, y1, color, thickness=1):
    """
    Draw a straight line between two points using Bresenham's algorithm.

    Parameters
    ----------
    image : np.ndarray
        Image array of shape (H, W) or (H, W, 3). Modified in place.
    x0 : int
        Column coordinate of the start point.
    y0 : int
        Row coordinate of the start point.
    x1 : int
        Column coordinate of the end point.
    y1 : int
        Row coordinate of the end point.
    color : int or tuple
        Grayscale scalar or RGB tuple (R, G, B). Values in [0, 255].
    thickness : int
        Thickness of the line in pixels. Default is 1.

    Returns
    -------
    np.ndarray
        The image with the line drawn (same array, modified in place).

    Raises
    ------
    TypeError
        If image is not a NumPy ndarray, coordinates are not ints,
        or color/thickness types are invalid.
    ValueError
        If color values are out of range or thickness is not positive.

    Notes
    -----
    Bresenham's line algorithm incrementally determines which pixels
    to fill to approximate a straight line between two points using
    only integer arithmetic. Points outside the canvas boundary are
    silently clipped via draw_point. Thickness is achieved by drawing
    a square of side=thickness at each step point along the line.
    """
    validate_image(image)
    _validate_color(color, image)
    _validate_thickness(thickness)
    for val, name in [(x0, "x0"), (y0, "y0"), (x1, "x1"), (y1, "y1")]:
        if not isinstance(val, int):
            raise TypeError(f"{name} must be an int, got {type(val).__name__}.")

    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy

    while True:
        half = thickness // 2
        _fill_rect(image, y0 - half, y0 - half + thickness,
                          x0 - half, x0 - half + thickness, color)
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy

    return image


# ---------------------------------------------------------------------------
# Rectangle
# ---------------------------------------------------------------------------

def draw_rectangle(image, x0, y0, x1, y1, color, thickness=1, filled=False):
    """
    Draw a rectangle on an image.

    Parameters
    ----------
    image : np.ndarray
        Image array of shape (H, W) or (H, W, 3). Modified in place.
    x0 : int
        Column coordinate of the top-left corner.
    y0 : int
        Row coordinate of the top-left corner.
    x1 : int
        Column coordinate of the bottom-right corner.
    y1 : int
        Row coordinate of the bottom-right corner.
    color : int or tuple
        Grayscale scalar or RGB tuple (R, G, B). Values in [0, 255].
    thickness : int
        Thickness of the outline in pixels. Default is 1.
        Ignored when filled=True.
    filled : bool
        If True, draws a solid filled rectangle.
        If False, draws only the outline. Default is False.

    Returns
    -------
    np.ndarray
        The image with the rectangle drawn (same array, modified in place).

    Raises
    ------
    TypeError
        If image is not a NumPy ndarray, coordinates are not ints,
        or color/thickness types are invalid.
    ValueError
        If color values are out of range, thickness is not positive,
        or x0 >= x1 or y0 >= y1.

    Notes
    -----
    All coordinates are clipped to canvas boundaries via _fill_rect.
    The filled rectangle uses _fill_rect directly — no pixel loops.
    The outline rectangle draws four lines using draw_line.
    """
    validate_image(image)
    _validate_color(color, image)
    _validate_thickness(thickness)
    for val, name in [(x0, "x0"), (y0, "y0"), (x1, "x1"), (y1, "y1")]:
        if not isinstance(val, int):
            raise TypeError(f"{name} must be an int, got {type(val).__name__}.")
    if x0 >= x1:
        raise ValueError(f"x0 ({x0}) must be less than x1 ({x1}).")
    if y0 >= y1:
        raise ValueError(f"y0 ({y0}) must be less than y1 ({y1}).")

    if filled:
        _fill_rect(image, y0, y1 + 1, x0, x1 + 1, color)
    else:
        draw_line(image, x0, y0, x1, y0, color, thickness)  # top
        draw_line(image, x0, y1, x1, y1, color, thickness)  # bottom
        draw_line(image, x0, y0, x0, y1, color, thickness)  # left
        draw_line(image, x1, y0, x1, y1, color, thickness)  # right

    return image


# ---------------------------------------------------------------------------
# Polygon
# ---------------------------------------------------------------------------

def draw_polygon(image, points, color, thickness=1, filled=False):
    """
    Draw a polygon defined by a list of vertices on an image.

    Parameters
    ----------
    image : np.ndarray
        Image array of shape (H, W) or (H, W, 3). Modified in place.
    points : list of tuple
        List of (x, y) integer coordinate pairs defining the polygon vertices.
        At least 3 points are required.
    color : int or tuple
        Grayscale scalar or RGB tuple (R, G, B). Values in [0, 255].
    thickness : int
        Thickness of the outline in pixels. Default is 1.
        Ignored when filled=True.
    filled : bool
        If True, draws a filled polygon using scanline fill.
        If False, draws only the outline. Default is False.

    Returns
    -------
    np.ndarray
        The image with the polygon drawn (same array, modified in place).

    Raises
    ------
    TypeError
        If image is not a NumPy ndarray, points is not a list,
        or color/thickness types are invalid.
    ValueError
        If fewer than 3 points are provided, or color values are out of range.

    Notes
    -----
    Outline: connects consecutive vertices with draw_line and closes
    the polygon by connecting the last vertex back to the first.
    Filled: uses a scanline algorithm. For each row in the bounding box,
    finds all edge intersections and fills between pairs. All fill
    operations go through _fill_rect which handles canvas clipping.
    """
    validate_image(image)
    _validate_color(color, image)
    _validate_thickness(thickness)
    if not isinstance(points, (list, tuple)):
        raise TypeError(f"points must be a list of (x, y) tuples, got {type(points).__name__}.")
    if len(points) < 3:
        raise ValueError(f"At least 3 points required, got {len(points)}.")

    if not filled:
        n = len(points)
        for i in range(n):
            x0, y0 = points[i]
            x1, y1 = points[(i + 1) % n]
            draw_line(image, int(x0), int(y0), int(x1), int(y1), color, thickness)
    else:
        H, W = image.shape[:2]
        ys = [p[1] for p in points]
        y_min = max(0, int(min(ys)))
        y_max = min(H - 1, int(max(ys)))
        n = len(points)

        for y in range(y_min, y_max + 1):
            intersections = []
            for i in range(n):
                x0, y0 = points[i]
                x1, y1 = points[(i + 1) % n]
                if y0 == y1:
                    continue
                if min(y0, y1) <= y < max(y0, y1):
                    x_intersect = x0 + (y - y0) * (x1 - x0) / (y1 - y0)
                    intersections.append(x_intersect)
            intersections.sort()
            for k in range(0, len(intersections) - 1, 2):
                c0 = int(np.ceil(intersections[k]))
                c1 = int(np.floor(intersections[k + 1]))
                _fill_rect(image, y, y + 1, c0, c1 + 1, color)

    return image


# ---------------------------------------------------------------------------
# Text
# ---------------------------------------------------------------------------

def draw_text(image, text, x, y, color, scale=1):
    """
    Draw ASCII text on an image using a built-in 5x7 bitmap font.

    Parameters
    ----------
    image : np.ndarray
        Image array of shape (H, W) or (H, W, 3). Modified in place.
    text : str
        String to draw. Supports printable ASCII characters.
    x : int
        Column coordinate of the top-left corner of the first character.
    y : int
        Row coordinate of the top-left corner of the first character.
    color : int or tuple
        Grayscale scalar or RGB tuple (R, G, B). Values in [0, 255].
    scale : int
        Font scale factor. Must be a positive integer.
        scale=1 renders each character at 5x7 pixels.
        scale=2 renders at 10x14 pixels, and so on.

    Returns
    -------
    np.ndarray
        The image with the text drawn (same array, modified in place).

    Raises
    ------
    TypeError
        If image is not a NumPy ndarray, text is not a str,
        x/y are not ints, or scale is not an int.
    ValueError
        If scale is not a positive integer, or color is out of range.

    Notes
    -----
    Uses a hardcoded 5x7 pixel bitmap font for a subset of ASCII characters.
    Each character is a list of 7 rows, each row being a 5-bit integer
    where bit 4 is the leftmost pixel and bit 0 is the rightmost.
    All pixel placement goes through _fill_rect for safe canvas clipping.
    Characters not in the font are skipped silently.
    """
    validate_image(image)
    _validate_color(color, image)
    if not isinstance(text, str):
        raise TypeError(f"text must be a str, got {type(text).__name__}.")
    if not isinstance(x, int) or not isinstance(y, int):
        raise TypeError(f"x and y must be ints, got {type(x).__name__} and {type(y).__name__}.")
    if not isinstance(scale, int):
        raise TypeError(f"scale must be an int, got {type(scale).__name__}.")
    if scale <= 0:
        raise ValueError(f"scale must be a positive integer, got {scale}.")

    FONT = {
        'A': [0x0E,0x11,0x11,0x1F,0x11,0x11,0x11],
        'B': [0x1E,0x11,0x11,0x1E,0x11,0x11,0x1E],
        'C': [0x0E,0x11,0x10,0x10,0x10,0x11,0x0E],
        'D': [0x1E,0x09,0x09,0x09,0x09,0x09,0x1E],
        'E': [0x1F,0x10,0x10,0x1E,0x10,0x10,0x1F],
        'F': [0x1F,0x10,0x10,0x1E,0x10,0x10,0x10],
        'G': [0x0E,0x11,0x10,0x17,0x11,0x11,0x0E],
        'H': [0x11,0x11,0x11,0x1F,0x11,0x11,0x11],
        'I': [0x0E,0x04,0x04,0x04,0x04,0x04,0x0E],
        'J': [0x01,0x01,0x01,0x01,0x01,0x11,0x0E],
        'K': [0x11,0x12,0x14,0x18,0x14,0x12,0x11],
        'L': [0x10,0x10,0x10,0x10,0x10,0x10,0x1F],
        'M': [0x11,0x1B,0x15,0x11,0x11,0x11,0x11],
        'N': [0x11,0x19,0x15,0x13,0x11,0x11,0x11],
        'O': [0x0E,0x11,0x11,0x11,0x11,0x11,0x0E],
        'P': [0x1E,0x11,0x11,0x1E,0x10,0x10,0x10],
        'Q': [0x0E,0x11,0x11,0x11,0x15,0x12,0x0D],
        'R': [0x1E,0x11,0x11,0x1E,0x14,0x12,0x11],
        'S': [0x0E,0x11,0x10,0x0E,0x01,0x11,0x0E],
        'T': [0x1F,0x04,0x04,0x04,0x04,0x04,0x04],
        'U': [0x11,0x11,0x11,0x11,0x11,0x11,0x0E],
        'V': [0x11,0x11,0x11,0x11,0x11,0x0A,0x04],
        'W': [0x11,0x11,0x11,0x15,0x15,0x1B,0x11],
        'X': [0x11,0x11,0x0A,0x04,0x0A,0x11,0x11],
        'Y': [0x11,0x11,0x0A,0x04,0x04,0x04,0x04],
        'Z': [0x1F,0x01,0x02,0x04,0x08,0x10,0x1F],
        '0': [0x0E,0x11,0x13,0x15,0x19,0x11,0x0E],
        '1': [0x04,0x0C,0x04,0x04,0x04,0x04,0x0E],
        '2': [0x0E,0x11,0x01,0x06,0x08,0x10,0x1F],
        '3': [0x1F,0x02,0x04,0x02,0x01,0x11,0x0E],
        '4': [0x02,0x06,0x0A,0x12,0x1F,0x02,0x02],
        '5': [0x1F,0x10,0x1E,0x01,0x01,0x11,0x0E],
        '6': [0x06,0x08,0x10,0x1E,0x11,0x11,0x0E],
        '7': [0x1F,0x01,0x02,0x04,0x08,0x08,0x08],
        '8': [0x0E,0x11,0x11,0x0E,0x11,0x11,0x0E],
        '9': [0x0E,0x11,0x11,0x0F,0x01,0x02,0x0C],
        ' ': [0x00,0x00,0x00,0x00,0x00,0x00,0x00],
        '.': [0x00,0x00,0x00,0x00,0x00,0x00,0x04],
        ',': [0x00,0x00,0x00,0x00,0x04,0x04,0x08],
        '!': [0x04,0x04,0x04,0x04,0x04,0x00,0x04],
        '?': [0x0E,0x11,0x01,0x06,0x04,0x00,0x04],
        '-': [0x00,0x00,0x00,0x1F,0x00,0x00,0x00],
        '_': [0x00,0x00,0x00,0x00,0x00,0x00,0x1F],
        ':': [0x00,0x04,0x00,0x00,0x00,0x04,0x00],
        '/': [0x01,0x02,0x02,0x04,0x08,0x08,0x10],
    }

    CHAR_W  = 5
    CHAR_H  = 7
    SPACING = 1

    cursor_x = x
    for char in text.upper():
        bitmap = FONT.get(char)
        if bitmap is None:
            cursor_x += (CHAR_W + SPACING) * scale
            continue

        for row_idx, row_bits in enumerate(bitmap):
            for col_idx in range(CHAR_W):
                bit = (row_bits >> (CHAR_W - 1 - col_idx)) & 1
                if bit:
                    px = cursor_x + col_idx * scale
                    py = y + row_idx * scale
                    _fill_rect(image, py, py + scale, px, px + scale, color)

        cursor_x += (CHAR_W + SPACING) * scale

    return image