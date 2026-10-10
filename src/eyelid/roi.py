"""Region of interest. CP-AnemiC images are already cut to the conjunctiva on a transparent background,
so the region is simply the fully opaque pixels; partly transparent edge pixels are blended and left out."""
import numpy as np
from PIL import Image


def roi_pixels(path, alpha_threshold=255):
    """RGB values in [0, 1] of the pixels inside the region, shape (n_pixels, 3)."""
    rgba = np.asarray(Image.open(path).convert("RGBA"), dtype=np.float32) / 255.0
    inside = rgba[..., 3] >= alpha_threshold / 255.0
    return rgba[inside][:, :3]
