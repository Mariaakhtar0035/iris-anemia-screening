"""Region of interest. CP-AnemiC images are already cut to the conjunctiva on a transparent background,
so the region is simply the fully opaque pixels; partly transparent edge pixels are blended and left out."""
import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError


def read_rgba(path, crop=False):
    """RGBA PIL image. `crop` trims to the non-transparent bounding box (Eyes-Defy masks are full frames)."""
    try:
        im = Image.open(path).convert("RGBA")
    except UnidentifiedImageError:
        # 187 Eyes-Defy masks carry a corrupt colour-profile (iCCP) chunk that PIL rejects; OpenCV reads the pixels.
        bgra = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if bgra is None:
            raise
        if bgra.dtype == np.uint16:  # some are 16 bits per channel; keep the top 8 bits, as PIL's RGBA conversion does
            bgra = (bgra >> 8).astype(np.uint8)
        im = Image.fromarray(cv2.cvtColor(bgra, cv2.COLOR_BGRA2RGBA))
    rgba = np.asarray(im)
    white = (rgba[..., :3] == 255).all(axis=-1)
    if (rgba[..., 3] == 255).all() and white.mean() > 0.5:
        # 31 Eyes-Defy masks (Italy folders 1-31) have no transparency: the cut-out sits on pure white. Make it transparent.
        rgba = rgba.copy()
        rgba[white, 3] = 0
        im = Image.fromarray(rgba, "RGBA")
    return im.crop(im.getchannel("A").getbbox()) if crop else im


def roi_pixels(path, alpha_threshold=255):
    """RGB values in [0, 1] of the pixels inside the region, shape (n_pixels, 3)."""
    rgba = np.asarray(read_rgba(path), dtype=np.float32) / 255.0
    inside = rgba[..., 3] >= alpha_threshold / 255.0
    return rgba[inside][:, :3]
