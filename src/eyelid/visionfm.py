"""Frozen VisionFM external-eye features (mentor advice M10, docs/eyelid_followup.md), cached per population.

VisionFM (github.com/ABILab-CUHK/VisionFM, research use only) is a ViT-B/16 pretrained with iBOT on ophthalmic images;
we use the teacher weights of its External Eye encoder. Features match the DINOv2 probe's: the CLS token and the mean
patch token of the last layer, concatenated.
"""
import argparse

import numpy as np
import timm
import torch

from src.common.io import ROOT
from src.eyelid.embed import pick_device, prepare

CACHE = ROOT / "data" / "processed" / "embeddings"


def load_encoder(cfg):
    # The checkpoint also pickles its training args and numpy scalars; allow only those, so no pickled code can run.
    safe = [argparse.Namespace, np.dtype, np.dtypes.Float64DType, np._core.multiarray.scalar,
            (np._core.multiarray.scalar, "numpy.core.multiarray.scalar")]  # the name numpy < 2 saved it under
    with torch.serialization.safe_globals(safe):
        ck = torch.load(ROOT / cfg["weights"], map_location="cpu", weights_only=True)
    state = {k.removeprefix("backbone."): v for k, v in ck["teacher"].items() if k.startswith("backbone.")}
    state.pop("masked_embed", None)  # iBOT's mask token, only used in pretraining
    model = timm.create_model(cfg["arch"], pretrained=False, num_classes=0)
    model.load_state_dict(state)  # strict: every ViT weight must be in the checkpoint
    return model.eval()


def embeddings(df, cfg, name, crop):
    """(n, 2 x width) CLS + mean patch token for each row of df, cached by population name and row ids."""
    path = CACHE / f"visionfm_external_{name}.npz"
    ids = df.id.astype(str).to_numpy()
    if path.exists():
        cached = np.load(path, allow_pickle=False)
        if list(cached["ids"]) == list(ids):
            return cached["cls_and_patch_mean"]
    device = pick_device()
    model = load_encoder(cfg).to(device)
    mean, std = np.array(cfg["mean"], np.float32), np.array(cfg["std"], np.float32)
    fill = tuple(int(round(255 * m)) for m in mean)  # background that normalizes to ~0
    rows = []
    for start in range(0, len(df), cfg["batch_size"]):
        paths = df.image_path.iloc[start:start + cfg["batch_size"]]
        x = (np.stack([np.asarray(prepare(ROOT / p, cfg["image_size"], fill, crop), np.float32) / 255.0
                       for p in paths]) - mean) / std
        with torch.no_grad():
            tokens = model.forward_features(torch.from_numpy(x).permute(0, 3, 1, 2).to(device))
        rows.append(torch.cat([tokens[:, 0], tokens[:, 1:].mean(1)], 1).cpu().numpy())
    out = np.concatenate(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, ids=ids, cls_and_patch_mean=out)
    return out
