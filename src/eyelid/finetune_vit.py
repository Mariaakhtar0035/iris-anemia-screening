"""Stage 9, biggest rung: fine-tune ViT-B/16 on each population's conjunctiva cut-outs (mentor advice M1).

5-fold stratified CV by photo with one seed (`vit` in configs/ceiling.yaml). Augmentation is applied in training
only: random flips, plus brightness and contrast jitter when jitter is on. Each (population, jitter) run is saved to
results/stage9_vit_<population>_<jitter|nojitter>.json, and src.eyelid.ceiling adds those runs to the Stage 9 report.

Run from the repo root:
  python -m src.eyelid.finetune_vit                                        # every run in the config
  python -m src.eyelid.finetune_vit --only eyes_defy_india --epochs 1      # smoke run, nothing saved
"""
import argparse
import time

import numpy as np
import torch
from torch import nn
from torchvision.transforms import v2
from transformers import AutoImageProcessor, ViTForImageClassification

from src.common.io import ROOT, load_config, save_results
from src.common.metrics import auroc
from src.common.splits import repeated_stratified
from src.eyelid.ceiling import population_frames
from src.eyelid.embed import pick_device, prepare


def load_images(df, size, crop):
    """Cut-outs on a mid-gray background, padded to a square: float tensor (n, 3, size, size) in [0, 1]."""
    return torch.stack([torch.from_numpy(np.asarray(prepare(ROOT / p, size, (128, 128, 128), crop), dtype=np.float32)
                                         / 255.0).permute(2, 0, 1) for p in df.image_path])


def predict(model, x, norm, device, batch=32):
    model.eval()
    with torch.no_grad():
        out = [torch.softmax(model(pixel_values=norm(x[i:i + batch]).to(device)).logits, -1)[:, 1].cpu()
               for i in range(0, len(x), batch)]
    return torch.cat(out).numpy()


def train_fold(x, y, cfg, jitter, seed, device, norm):
    torch.manual_seed(seed)  # also seeds the random augmentations
    model = ViTForImageClassification.from_pretrained(cfg["model_id"], revision=cfg["revision"], num_labels=2).to(device)
    opt = torch.optim.NAdam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    augment = v2.Compose([v2.RandomHorizontalFlip(), v2.RandomVerticalFlip()]
                         + ([v2.ColorJitter(**cfg["jitter"])] if jitter else []))
    loss_fn = nn.CrossEntropyLoss()
    rng = np.random.default_rng(seed)
    model.train()
    for _ in range(cfg["epochs"]):
        order = rng.permutation(len(y))
        for start in range(0, len(order), cfg["batch_size"]):
            b = order[start:start + cfg["batch_size"]]
            xb = torch.stack([augment(x[i]) for i in b])  # per image, so each gets its own random jitter
            loss = loss_fn(model(pixel_values=norm(xb).to(device)).logits, torch.as_tensor(y[b], device=device))
            opt.zero_grad()
            loss.backward()
            opt.step()
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", nargs="+", help="populations to run (default: all in the config)")
    parser.add_argument("--epochs", type=int, help="override epochs for a smoke run; nothing is saved")
    args = parser.parse_args()
    base, cfg = load_config("default"), load_config("ceiling")
    vit = cfg["vit"]
    if args.epochs:
        vit["epochs"] = args.epochs
    frames = population_frames(cfg)
    proc = AutoImageProcessor.from_pretrained(vit["model_id"], revision=vit["revision"])
    mean, std = torch.tensor(proc.image_mean)[:, None, None], torch.tensor(proc.image_std)[:, None, None]
    norm = lambda t: (t - mean) / std  # noqa: E731
    device = pick_device()

    for name, settings in vit["runs"].items():
        if args.only and name not in args.only:
            continue
        df = frames[name]
        y = df.anemic.to_numpy()
        x = load_images(df, vit["image_size"], crop=name != "cp_anemic")  # Eyes-Defy masks are full frames
        folds = repeated_stratified(y, vit["n_splits"], 1, base["seed"])
        for jitter in settings:
            t0, held_out, train_auroc = time.time(), np.full(len(y), np.nan), []
            for k, (_, train, test) in enumerate(folds):
                model = train_fold(x[train], y[train], vit, jitter, base["seed"] + k, device, norm)
                held_out[test] = predict(model, x[test], norm, device)
                train_auroc.append(auroc(y[train], predict(model, x[train], norm, device)))
                print(f"{name} jitter={jitter} fold {k}: train AUROC {train_auroc[-1]:.3f}, "
                      f"{time.time() - t0:.0f}s so far", flush=True)
                del model
            assert not np.isnan(held_out).any()
            print(f"{name} jitter={jitter}: held-out AUROC {auroc(y, held_out):.3f}", flush=True)
            if not args.epochs:
                save_results(f"stage9_vit_{name}_{'jitter' if jitter else 'nojitter'}",
                             {"population": name, "jitter": jitter, "device": device, "ids": df.id.tolist(),
                              "held_out": held_out.tolist(), "train_auroc": train_auroc,
                              "seconds": round(time.time() - t0)},
                             {"default": base, "ceiling": cfg})


if __name__ == "__main__":
    main()
