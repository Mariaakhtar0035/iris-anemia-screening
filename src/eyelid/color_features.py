"""Color statistics inside the region of interest."""
import cv2
import numpy as np
import pandas as pd

from src.eyelid.roi import roi_pixels

CHANNELS = ["R", "G", "B", "H", "S", "V", "L", "a", "b"]


def gray_world(rgb):
    """Scale each channel so the three channel means are equal."""
    means = rgb.mean(axis=0)
    return np.clip(rgb * (means.mean() / means), 0.0, 1.0)


def color_features(rgb, normalize=False):
    """29 features: mean, median and std of R G B H S V L a b, plus the R/G ratio and a - b."""
    if normalize:
        rgb = gray_world(rgb)
    img = rgb.reshape(-1, 1, 3).astype(np.float32)
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).reshape(-1, 3)  # H in degrees, S and V in [0, 1]
    # Conjunctiva hues sit on both sides of 0/360 degrees, so averaging raw hue is meaningless. Rotate red to 180.
    hsv[:, 0] = (hsv[:, 0] + 180.0) % 360.0
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB).reshape(-1, 3)  # L in [0, 100], a and b around 0
    pixels = np.hstack([rgb, hsv, lab])

    feats = {}
    for i, ch in enumerate(CHANNELS):
        feats[f"{ch}_mean"] = float(pixels[:, i].mean())
        feats[f"{ch}_median"] = float(np.median(pixels[:, i]))
        feats[f"{ch}_std"] = float(pixels[:, i].std())
    feats["R_over_G"] = feats["R_mean"] / max(feats["G_mean"], 1e-6)
    feats["a_minus_b"] = feats["a_mean"] - feats["b_mean"]
    return feats


def feature_table(image_paths, ids, normalize=False, alpha_threshold=255, min_pixels=50):
    rows = []
    for path in image_paths:
        rgb = roi_pixels(path, alpha_threshold)
        assert len(rgb) >= min_pixels, f"{path}: only {len(rgb)} pixels inside the region"
        rows.append(color_features(rgb, normalize))
    return pd.DataFrame(rows, index=pd.Index(ids, name="id"))


def image_properties(image_paths, ids):
    """Size and cut-out shape only, no color. A negative control: a model on these should sit near chance
    unless it can match up duplicate images."""
    from PIL import Image

    rows = []
    for path in image_paths:
        im = Image.open(path).convert("RGBA")
        alpha = np.asarray(im)[..., 3]
        w, h = im.size
        rows.append({
            "width": w, "height": h, "aspect": w / h,
            "opaque_frac": float((alpha == 255).mean()),
            "edge_frac": float(((alpha > 0) & (alpha < 255)).mean()),
        })
    return pd.DataFrame(rows, index=pd.Index(ids, name="id"))
