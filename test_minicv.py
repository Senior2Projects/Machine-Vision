import numpy as np
import sys

print("=" * 50)
print("minicv test suite")
print("=" * 50)

# ── utils ──────────────────────────────────────────
print("\n[1] utils")
from minicv.utils import normalize, clip_pixels, pad, convolve2d

img_gray = np.random.randint(0, 256, (64, 64), dtype=np.uint8)
img_rgb  = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)

assert normalize(img_gray, mode="minmax").max() <= 1.0
assert normalize(img_gray, mode="uint8").dtype == np.uint8
assert normalize(img_gray, mode="zscore").mean() < 0.5
assert clip_pixels(img_gray).min() >= 0
assert pad(img_gray, 2, 2, mode="constant").shape == (68, 68)
assert pad(img_gray, 2, 2, mode="reflect").shape  == (68, 68)
assert pad(img_gray, 2, 2, mode="replicate").shape == (68, 68)
kernel = np.ones((3, 3), dtype=np.float64) / 9
assert convolve2d(img_gray, kernel).shape == (64, 64)
print("  OK")

# ── io ─────────────────────────────────────────────
print("\n[2] io")
from minicv.io import read_image, save_image
import os

# Create a tiny synthetic image and save it
synthetic = np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8)
save_image(synthetic, "test_output.png")
assert os.path.exists("test_output.png")

loaded = read_image("test_output.png", mode="rgb")
assert loaded.shape == (32, 32, 3)

loaded_gray = read_image("test_output.png", mode="gray")
assert loaded_gray.ndim == 2
os.remove("test_output.png")
print("  OK")

# ── color ──────────────────────────────────────────
print("\n[3] color")
from minicv.color import rgb_to_gray, gray_to_rgb

gray = rgb_to_gray(img_rgb)
assert gray.shape == (64, 64)

back = gray_to_rgb(gray)
assert back.shape == (64, 64, 3)
print("  OK")

# ── filtering ──────────────────────────────────────
print("\n[4] filtering")
from minicv.filtering import (
    spatial_filter, mean_filter, gaussian_kernel,
    gaussian_filter, median_filter, sobel_gradients
)

assert mean_filter(img_gray, kernel_size=3).shape == (64, 64)
assert mean_filter(img_rgb,  kernel_size=3).shape == (64, 64, 3)

gk = gaussian_kernel(5, sigma=1.0)
assert gk.shape == (5, 5)
assert abs(gk.sum() - 1.0) < 1e-6

assert gaussian_filter(img_gray, kernel_size=5, sigma=1.0).shape == (64, 64)
assert median_filter(img_gray, kernel_size=3).shape  == (64, 64)
assert median_filter(img_rgb,  kernel_size=3).shape  == (64, 64, 3)

Gx, Gy, mag = sobel_gradients(img_gray)
assert Gx.shape == Gy.shape == mag.shape == (64, 64)
print("  OK")

# ── thresholding ───────────────────────────────────
print("\n[5] thresholding")
from minicv.thresholding import global_threshold, otsu_threshold, adaptive_threshold

gt = global_threshold(img_gray, threshold=128)
assert gt.shape == (64, 64)
assert set(np.unique(gt)).issubset({0, 255})

binary, t = otsu_threshold(img_gray)
assert binary.shape == (64, 64)
assert 0 <= t <= 255

at = adaptive_threshold(img_gray, block_size=11, C=2, method="mean")
assert at.shape == (64, 64)
at2 = adaptive_threshold(img_gray, block_size=11, C=2, method="gaussian")
assert at2.shape == (64, 64)
print("  OK")

# ── techniques ─────────────────────────────────────
print("\n[6] techniques")
from minicv.techniques import (
    bit_plane_slice, histogram,
    histogram_equalization, contrast_stretching,
    laplacian_sharpening
)

for plane in range(8):
    bp = bit_plane_slice(img_gray, plane)
    assert bp.shape == (64, 64)
    assert set(np.unique(bp)).issubset({0, 255})

counts, edges = histogram(img_gray, bins=256)
assert len(counts) == 256

eq = histogram_equalization(img_gray)
assert eq.shape == (64, 64)
assert eq.dtype == np.uint8

cs = contrast_stretching(img_gray)
assert cs.shape == (64, 64)

ls = laplacian_sharpening(img_gray)
assert ls.shape == (64, 64)
ls_rgb = laplacian_sharpening(img_rgb)
assert ls_rgb.shape == (64, 64, 3)
print("  OK")

# ── transforms ─────────────────────────────────────
print("\n[7] transforms")
from minicv.transforms import resize, rotate, translate

r1 = resize(img_gray, 32, 32, method="nearest")
assert r1.shape == (32, 32)
r2 = resize(img_gray, 32, 32, method="bilinear")
assert r2.shape == (32, 32)
r3 = resize(img_rgb, 32, 32, method="bilinear")
assert r3.shape == (32, 32, 3)

rot = rotate(img_gray, angle=45, method="bilinear")
assert rot.shape == (64, 64)
rot_rgb = rotate(img_rgb, angle=30, method="bilinear")
assert rot_rgb.shape == (64, 64, 3)

tr = translate(img_gray, shift_x=10, shift_y=5)
assert tr.shape == (64, 64)
print("  OK")

# ── features ───────────────────────────────────────
print("\n[8] features")
from minicv.features import (
    color_histogram_descriptor, hu_moments_descriptor,
    hog_descriptor, lbp_descriptor
)

ch = color_histogram_descriptor(img_rgb, bins=32)
assert ch.shape == (96,)
assert abs(ch[:32].sum() - 1.0) < 1e-6

hu = hu_moments_descriptor(img_gray)
assert hu.shape == (7,)

hog = hog_descriptor(img_gray, cell_size=8, bins=9)
assert hog.ndim == 1

lbp = lbp_descriptor(img_gray, radius=1, n_points=8, bins=256)
assert lbp.shape == (256,)
assert abs(lbp.sum() - 1.0) < 1e-6
print("  OK")

# ── drawing ────────────────────────────────────────
print("\n[9] drawing")
from minicv.drawing import (
    draw_point, draw_line, draw_rectangle,
    draw_polygon, draw_text
)

canvas_gray = np.zeros((128, 128), dtype=np.uint8)
canvas_rgb  = np.zeros((128, 128, 3), dtype=np.uint8)

draw_point(canvas_gray, 64, 64, color=255, thickness=3)
draw_point(canvas_rgb,  64, 64, color=(255, 0, 0), thickness=3)

draw_line(canvas_gray, 0, 0, 127, 127, color=255)
draw_line(canvas_rgb,  0, 0, 127, 127, color=(0, 255, 0))

draw_rectangle(canvas_gray, 10, 10, 50, 50, color=255, filled=False)
draw_rectangle(canvas_rgb,  10, 10, 50, 50, color=(0, 0, 255), filled=True)

pts = [(20, 60), (60, 40), (100, 60), (80, 100), (40, 100)]
draw_polygon(canvas_gray, pts, color=255, filled=False)
draw_polygon(canvas_rgb,  pts, color=(255, 255, 0), filled=True)

draw_text(canvas_gray, "HELLO", x=5, y=5, color=255, scale=1)
draw_text(canvas_rgb,  "TEST",  x=5, y=5, color=(255, 255, 255), scale=2)
print("  OK")

# ── summary ────────────────────────────────────────
print("\n" + "=" * 50)
print("All tests passed.")
print("=" * 50)