"""Replicate the CP-AnemiC paper's joint ResNet50 benchmark under its own split and under a split by photo.

Run from the repo root: python -m src.eyelid.replicate_cnn [--epochs N] [--folds K]  (flags only for timing)
"""
import argparse
import time

import numpy as np
import torch
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from torch import nn
from torchvision import models
from torchvision.transforms import v2

from src.common.evaluate import summarize
from src.common.io import ROOT, load_config, save_results
from src.eyelid.dataset import load_cp_anemic
from src.eyelid.embed import pick_device, prepare
from src.eyelid.report import markdown_table

IMAGENET_MEAN, IMAGENET_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
FILL = tuple(int(round(255 * m)) for m in IMAGENET_MEAN)


class JointNet(nn.Module):
    """Backbone + two heads: anemia logit and standardized Hb, as in the paper's Fig. 4."""

    def __init__(self, pretrained):
        super().__init__()
        self.backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
        width = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.cls_head, self.hb_head = nn.Linear(width, 1), nn.Linear(width, 1)

    def forward(self, x):
        f = self.backbone(x)
        return self.cls_head(f).squeeze(1), self.hb_head(f).squeeze(1)


def load_images(paths, size):
    return torch.stack([v2.functional.pil_to_tensor(prepare(ROOT / p, size, FILL)) for p in paths])  # uint8


def train_fold(images, y, hb_z, train_idx, cfg, device, seed):
    torch.manual_seed(seed)
    model = JointNet(cfg["pretrained"]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    a = cfg["augment"]
    augment = v2.Compose([
        v2.RandomHorizontalFlip(a["hflip"]),
        v2.RandomAffine(degrees=a["degrees"], translate=(a["translate"],) * 2, scale=tuple(a["scale"]), fill=list(FILL)),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    bce, mse = nn.BCEWithLogitsLoss(), nn.MSELoss()
    rng = np.random.default_rng(seed)
    model.train()
    for _ in range(cfg["epochs"]):
        order = rng.permutation(train_idx)
        for start in range(0, len(order), cfg["batch_size"]):
            b = order[start:start + cfg["batch_size"]]
            x = torch.stack([augment(images[i]) for i in b]).to(device)
            logit, hb_pred = model(x)
            loss = bce(logit, torch.as_tensor(y[b], dtype=torch.float32, device=device)) + mse(
                hb_pred, torch.as_tensor(hb_z[b], dtype=torch.float32, device=device))
            opt.zero_grad()
            loss.backward()
            opt.step()
    return model


@torch.no_grad()
def predict(model, images, idx, device, batch_size):
    model.eval()
    to_input = v2.Compose([v2.ToDtype(torch.float32, scale=True), v2.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
    probs, hbs = [], []
    for start in range(0, len(idx), batch_size):
        x = torch.stack([to_input(images[i]) for i in idx[start:start + batch_size]]).to(device)
        logit, hb = model(x)
        probs.append(torch.sigmoid(logit).cpu().numpy())
        hbs.append(hb.cpu().numpy())
    return np.concatenate(probs), np.concatenate(hbs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--folds", type=int, help="run only the first K folds (timing)")
    args = parser.parse_args()

    base, cfg = load_config("default"), load_config("replicate")
    if args.epochs:
        cfg["epochs"] = args.epochs
    seed, boot = base["seed"], base["bootstrap"]
    device = pick_device()
    data = load_cp_anemic("all")  # the paper used every row
    y, hb, groups = data.anemic.to_numpy(), data.hb.to_numpy(), data.photo_group.to_numpy()
    images = load_images(data.image_path, cfg["image_size"])

    splitters = {
        "paper_random_rows": StratifiedKFold(cfg["n_splits"], shuffle=True, random_state=seed).split(images, y),
        "grouped_by_photo": StratifiedGroupKFold(cfg["n_splits"], shuffle=True, random_state=seed).split(images, y, groups),
    }
    rows = []
    for split_name in cfg["splits"]:
        prob, hb_pred = np.full(len(y), np.nan), np.full(len(y), np.nan)
        for k, (tr, te) in enumerate(splitters[split_name]):
            if args.folds and k >= args.folds:
                break
            t0 = time.time()
            mu, sd = hb[tr].mean(), hb[tr].std()  # Hb standardized with training-fold statistics only
            model = train_fold(images, y, (hb - mu) / sd, tr, cfg, device, seed + k)
            p, z = predict(model, images, te, device, cfg["batch_size"])
            prob[te], hb_pred[te] = p, z * sd + mu
            print(f"{split_name} fold {k + 1}: {time.time() - t0:.0f}s on {device}", flush=True)
        done = ~np.isnan(prob)
        if args.folds:
            print(f"timing run only: {done.sum()} rows scored")
            continue
        err = hb_pred - hb
        rows.append({
            "split": split_name, "n": len(y), "model": "joint_resnet50",
            **summarize(y, prob, boot["n_resamples"], boot["ci"], seed, groups=groups),
            "accuracy_at_0.5": float(((prob >= 0.5) == y).mean()),
            "hb_mae": float(np.abs(err).mean()), "hb_mse": float((err ** 2).mean()),
        })
    if args.folds:
        return

    path = save_results("replicate_cnn", {"device": device, "rows": rows}, {"default": base, "replicate": cfg})
    import json
    git = json.loads(path.read_text())["git"]
    commit = git["commit"][:7] + (" + uncommitted changes" if git["dirty"] else "")
    table = markdown_table(rows, ["split", "n", "model"])
    extra = "\n".join(f"| {r['split']} | {r['accuracy_at_0.5']:.1%} | {r['hb_mae']:.2f} | {r['hb_mse']:.2f} |" for r in rows)
    (ROOT / "reports" / "replicate_cnn.md").write_text(
        "# Replication: the CP-AnemiC paper's joint ResNet50\n\n"
        f"Generated by `python -m src.eyelid.replicate_cnn` at commit `{commit}` on `{device}`. Settings and the "
        "deviations from the paper are in `configs/replicate.yaml`. Anemia = Hb < 11.0 g/dL. All 710 rows, as in "
        "the paper. Bootstrap 95% CIs (2000 resamples) resample whole photo groups.\n\n"
        "- `paper_random_rows`: stratified 5-fold CV over rows, the paper's protocol. Copies of a photo can sit "
        "in both train and test.\n- `grouped_by_photo`: identical, except every copy of a photo stays in one fold.\n\n"
        f"{table}\n\nFor comparison with the paper's Tables 3 and 4 (accuracy and Hb error are not threshold-free; "
        "reported only alongside the metrics above):\n\n| split | Accuracy at 0.5 | Hb MAE (g/dL) | Hb MSE |\n|---|---|---|---|\n"
        f"{extra}\n\nPaper (Appiahene et al. 2023, Tables 3-4): ResNet50 AUC 0.835, accuracy 84.79%, Hb MAE 1.54, "
        "MSE 4.15; ViT AUC 0.841, accuracy 84.08%, MAE 1.50, MSE 3.82.\n"
    )
    print(f"\n{table}\n\n{extra}\n\nsaved {path.relative_to(ROOT)} and reports/replicate_cnn.md")


if __name__ == "__main__":
    main()
