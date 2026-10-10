"""Replicate the 2025 ViT paper's 98.47% on Eyes-Defy-Anemia, under its own protocol and with the split by photo.

Run from the repo root:
  python -m src.eyelid.replicate_vit                  # seeds from the config
  python -m src.eyelid.replicate_vit --seeds 1 2 3 4  # more seeds, each saved to its own results file
  python -m src.eyelid.replicate_vit --report         # rebuild reports/replicate_vit.md from every seed run
  python -m src.eyelid.replicate_vit --epochs 1 --timing
"""
import argparse
import json
import time

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from torch import nn
from transformers import AutoImageProcessor, ViTForImageClassification

from src.common.evaluate import summarize
from src.common.io import RESULTS, ROOT, load_config, save_results
from src.eyelid.embed import pick_device
from src.eyelid.report import markdown_table

LABELS = ROOT / "data" / "processed" / "eyes_defy_labels.csv"
VERSIONS = ("original", "hflip", "vflip")


def load_photos(labels, size, mean, std):
    """Every photo resized to size x size and normalized, plus its two flips: tensor (n_photos, 3, 3, H, W)."""
    out = []
    for p in labels.image_path:
        x = torch.from_numpy(np.asarray(Image.open(ROOT / p).convert("RGB").resize((size, size), Image.BICUBIC),
                                        dtype=np.float32) / 255.0).permute(2, 0, 1)
        x = (x - torch.tensor(mean)[:, None, None]) / torch.tensor(std)[:, None, None]
        out.append(torch.stack([x, torch.flip(x, dims=[2]), torch.flip(x, dims=[1])]))
    return torch.stack(out)


def make_split(protocol, n_photos, seed, test_size):
    """Return (train, test) as arrays of (photo, version) pairs."""
    every = np.array([(p, v) for p in range(n_photos) for v in range(len(VERSIONS))])
    if protocol == "paper":  # augment first, then split the 3 * n images at random
        train, test = train_test_split(every, test_size=test_size, random_state=seed, shuffle=True)
    elif protocol == "by_photo":  # split photos first; all versions follow their photo
        train_p, test_p = train_test_split(np.arange(n_photos), test_size=test_size, random_state=seed, shuffle=True)
        train, test = every[np.isin(every[:, 0], train_p)], every[np.isin(every[:, 0], test_p)]
    else:
        raise ValueError(protocol)
    return train, test


def run(protocol, seed, images, y, cfg, device):
    train, test = make_split(protocol, len(y), seed, cfg["test_size"])
    shared = len(set(train[:, 0]) & set(test[:, 0]))
    torch.manual_seed(seed)
    model = ViTForImageClassification.from_pretrained(cfg["model_id"], revision=cfg["revision"], num_labels=2).to(device)
    opt = torch.optim.NAdam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    loss_fn = nn.CrossEntropyLoss()
    rng = np.random.default_rng(seed)
    t0 = time.time()
    model.train()
    for _ in range(cfg["epochs"]):
        order = train[rng.permutation(len(train))]
        for start in range(0, len(order), cfg["batch_size"]):
            b = order[start:start + cfg["batch_size"]]
            x = images[b[:, 0], b[:, 1]].to(device)
            loss = loss_fn(model(pixel_values=x).logits, torch.as_tensor(y[b[:, 0]], device=device))
            opt.zero_grad()
            loss.backward()
            opt.step()
    model.eval()
    probs = []
    with torch.no_grad():
        for start in range(0, len(test), 32):
            b = test[start:start + 32]
            probs.append(torch.softmax(model(pixel_values=images[b[:, 0], b[:, 1]].to(device)).logits, -1)[:, 1].cpu().numpy())
    prob = np.concatenate(probs)
    y_test, pred = y[test[:, 0]], (prob >= 0.5).astype(int)
    report = classification_report(y_test, pred, labels=[0, 1], target_names=["no_anemia", "anemia"],
                                   output_dict=True, zero_division=0)
    return {
        "protocol": protocol, "seed": seed, "n_train_images": len(train), "n_test_images": len(test),
        "test_photos_also_in_train": shared, "train_seconds": round(time.time() - t0),
        "accuracy": report["accuracy"], "per_class": {k: report[k] for k in ("no_anemia", "anemia")},
        "test_photo_ids": sorted(set(test[:, 0].tolist())),
        "y_test": y_test.tolist(), "prob": prob.tolist(), "test_photo_index": test[:, 0].tolist(),
    }


def build_report(base):
    runs = []
    for path in sorted(RESULTS.glob("replicate_vit_seed*.json")):
        d = json.loads(path.read_text())
        runs += [{**r, "commit": d["git"]["commit"][:7] + ("+dirty" if d["git"]["dirty"] else "")} for r in d["results"]["runs"]]
    if not runs:
        raise SystemExit("no replicate_vit_seed*.json results yet")
    boot = base["bootstrap"]
    rows = []
    for r in runs:
        y, p, g = np.array(r["y_test"]), np.array(r["prob"]), np.array(r["test_photo_index"])
        m = summarize(y, p, boot["n_resamples"], boot["ci"], base["seed"], groups=g)  # resample whole photos
        rows.append({**r, **m, "split": f"{r['protocol']} (seed {r['seed']})", "n": r["n_test_images"]})
    table = markdown_table(rows, ["split", "n"])
    lines = [f"| {r['protocol']} | {r['seed']} | {r['accuracy']:.2%} | {r['per_class']['anemia']['precision']:.3f} | "
             f"{r['per_class']['anemia']['recall']:.3f} | {r['per_class']['no_anemia']['recall']:.3f} | "
             f"{r['test_photos_also_in_train']} | {r['commit']} |" for r in rows]
    acc = pd.DataFrame(rows).groupby("protocol").accuracy.agg(["mean", "min", "max", "count"])
    acc_lines = [f"| {p} | {a['mean']:.2%} | {a['min']:.2%} | {a['max']:.2%} | {int(a['count'])} |" for p, a in acc.iterrows()]
    text = (
        "# Replication: the 2025 ViT paper (98.47% on Eyes-Defy-Anemia)\n\n"
        "Generated by `python -m src.eyelid.replicate_vit --report`. Settings in `configs/replicate_vit.yaml`. "
        "Labels use the paper's cutoffs (India women < 12, men < 14 g/dL; Italy < 10.5 g/dL). Bootstrap 95% CIs "
        "(2000 resamples) resample whole photos.\n\n"
        "- `paper`: every photo flipped twice, then the 3x images split 80/20 at random (the paper's protocol).\n"
        "- `by_photo`: the photos split 80/20 first; all three versions of a test photo stay in test. Same test "
        "make-up, but no photo is on both sides.\n\n"
        f"{table}\n\nThe paper's own metrics, for comparison with its Table 2 (ViT-21k: 98.47% accuracy):\n\n"
        "| protocol | seed | accuracy | anemia precision | anemia recall | no-anemia recall | test photos also in train | commit |\n"
        "|---|---|---|---|---|---|---|---|\n" + "\n".join(lines) + "\n\nAccuracy across seeds:\n\n"
        + "| protocol | mean | min | max | seeds |\n|---|---|---|---|---|\n" + "\n".join(acc_lines) + "\n"
    )
    (ROOT / "reports" / "replicate_vit.md").write_text(text)
    print(text)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--timing", action="store_true", help="one protocol, nothing saved")
    parser.add_argument("--report", action="store_true")
    args = parser.parse_args()
    base, cfg = load_config("default"), load_config("replicate_vit")
    if args.report:
        return build_report(base)
    if args.epochs:
        cfg["epochs"] = args.epochs

    labels = pd.read_csv(LABELS)
    labels = labels[labels.anemic_paper.notna()].reset_index(drop=True)
    y = labels.anemic_paper.astype(int).to_numpy()
    proc = AutoImageProcessor.from_pretrained(cfg["model_id"], revision=cfg["revision"])
    images = load_photos(labels, cfg["image_size"], proc.image_mean, proc.image_std)
    device = pick_device()
    print(f"{len(y)} photos ({y.sum()} anemic) -> {images.shape[0] * images.shape[1]} images; device {device}", flush=True)

    for seed in args.seeds or cfg["seeds"]:
        runs = []
        for protocol in cfg["protocols"][:1] if args.timing else cfg["protocols"]:
            r = run(protocol, seed, images, y, cfg, device)
            print(f"seed {seed} {protocol}: accuracy {r['accuracy']:.2%}, {r['test_photos_also_in_train']} test photos "
                  f"also in train, {r['train_seconds']}s", flush=True)
            runs.append(r)
        if not args.timing:
            save_results(f"replicate_vit_seed{seed}", {"device": device, "photo_ids": labels.id.tolist(), "runs": runs},
                         {"default": base, "replicate_vit": cfg})
    if not args.timing:
        build_report(base)


if __name__ == "__main__":
    main()
