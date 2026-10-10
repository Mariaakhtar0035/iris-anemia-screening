"""Frozen DINOv2 embeddings for every CP-AnemiC image, cached so they are computed once.

Run from the repo root: python -m src.eyelid.embed
"""
import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

from src.common.io import ROOT, load_config
from src.eyelid.dataset import load_cp_anemic

CACHE = ROOT / "data" / "processed" / "embeddings" / "dinov2_small.npz"


def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def prepare(path, size, fill):
    """Composite the cut-out conjunctiva on `fill`, pad to a square with it, resize to size x size."""
    im = Image.open(path).convert("RGBA")
    flat = Image.alpha_composite(Image.new("RGBA", im.size, fill + (255,)), im).convert("RGB")
    side = max(flat.size)
    square = Image.new("RGB", (side, side), fill)
    square.paste(flat, ((side - flat.width) // 2, (side - flat.height) // 2))
    return square.resize((size, size), Image.BICUBIC)


def load_embeddings(cfg):
    """Cached embeddings as a dict of arrays keyed by feature set, plus ids. Recomputes if the cache is stale."""
    labels = load_cp_anemic("all")
    if CACHE.exists():
        cached = np.load(CACHE, allow_pickle=False)
        if str(cached["revision"]) == cfg["revision"] and list(cached["ids"]) == list(labels.id):
            return {k: cached[k] for k in cached.files}
    return compute(labels, cfg)


def compute(labels, cfg):
    device = pick_device()
    processor = AutoImageProcessor.from_pretrained(cfg["model_id"], revision=cfg["revision"])
    model = AutoModel.from_pretrained(cfg["model_id"], revision=cfg["revision"]).to(device).eval()
    mean = np.array(processor.image_mean, dtype=np.float32)
    std = np.array(processor.image_std, dtype=np.float32)
    fill = tuple(int(round(255 * m)) for m in mean)  # background that normalizes to ~0

    cls_rows, patch_rows = [], []
    paths = [ROOT / p for p in labels.image_path]
    for start in range(0, len(paths), cfg["batch_size"]):
        batch = [prepare(p, cfg["image_size"], fill) for p in paths[start:start + cfg["batch_size"]]]
        x = (np.stack([np.asarray(im, dtype=np.float32) / 255.0 for im in batch]) - mean) / std
        with torch.no_grad():
            hidden = model(pixel_values=torch.from_numpy(x).permute(0, 3, 1, 2).to(device)).last_hidden_state
        cls_rows.append(hidden[:, 0].cpu().numpy())
        patch_rows.append(hidden[:, 1:].mean(dim=1).cpu().numpy())
        print(f"embedded {min(start + cfg['batch_size'], len(paths))}/{len(paths)} on {device}")

    cls, patch_mean = np.concatenate(cls_rows), np.concatenate(patch_rows)
    out = {
        "ids": labels.id.to_numpy().astype(str),
        "cls": cls,
        "patch_mean": patch_mean,
        "cls_and_patch_mean": np.hstack([cls, patch_mean]),
        "revision": np.array(cfg["revision"]),
        "device": np.array(device),
    }
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez(CACHE, **out)
    return out


if __name__ == "__main__":
    emb = load_embeddings(load_config("probe"))
    print(f"cache: {CACHE.relative_to(ROOT)}, device used: {emb['device']}, shapes: "
          + ", ".join(f"{k} {emb[k].shape}" for k in ["cls", "patch_mean", "cls_and_patch_mean"]))
