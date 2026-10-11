"""Figures for the README, drawn from results/*.json.

Run from the repo root: python -m src.figures
"""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.common.io import RESULTS, ROOT

OUT = ROOT / "reports" / "figures"
# Published numbers, typed from the papers: Appiahene et al. 2023 Table 3; Ramos-Soto et al. 2025 Table 2.
PUBLISHED = {"CP-AnemiC\njoint ResNet50": 0.8479, "Eyes-Defy\nViT-B/16": 0.9847}
VIT = "ViT-B/16 in21k"


def load(name):
    return json.loads((RESULTS / f"{name}.json").read_text())["results"]


def replication_bars():
    cnn = {r["split"]: r["accuracy_at_0.5"] for r in load("replicate_cnn")["rows"]}
    vit = load("replicate_vit")
    bars = {
        "published": list(PUBLISHED.values()),
        "ours, paper's split": [cnn["paper_random_rows"], vit["paper"][VIT]["accuracy"]],
        "ours, no photo in both train and test": [cnn["grouped_by_photo"], vit["by_photo"][VIT]["accuracy"]],
        "country rule (India = anemic)": [np.nan, vit["by_photo"]["country rule (India = anemic)"]["accuracy"]],
    }
    x = np.arange(len(PUBLISHED))
    w = 0.2
    fig, ax = plt.subplots(figsize=(7.5, 4))
    for i, (label, vals) in enumerate(bars.items()):
        b = ax.bar(x + (i - 1.5) * w, vals, w, label=label)
        ax.bar_label(b, labels=["" if np.isnan(v) else f"{v:.1%}" for v in vals], fontsize=8, padding=2)
    ax.set_xticks(x, PUBLISHED.keys())
    ax.set_ylim(0.5, 1.05)
    ax.set_ylabel("test accuracy")
    ax.set_title("Published vs replicated accuracy")
    ax.legend(fontsize=8, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "replication.png", dpi=150)


def epoch_curves():
    paths = sorted(RESULTS.glob("replicate_vit_epochs_seed*.json"))
    if not paths:
        return
    fig, ax = plt.subplots(figsize=(7.5, 4))
    for protocol, color in (("paper", "C1"), ("by_photo", "C2")):
        curves = np.array([r["test_accuracy_by_epoch"] for p in paths
                           for r in json.loads(p.read_text())["results"]["runs"] if r["protocol"] == protocol])
        epochs = np.arange(1, curves.shape[1] + 1)
        for c in curves:
            ax.plot(epochs, c, color=color, alpha=0.25, lw=1)
        ax.plot(epochs, curves.mean(0), color=color, lw=2,
                label=f"{'paper split' if protocol == 'paper' else 'no photo in both'} (mean of {len(curves)} seeds)")
    ax.axhline(PUBLISHED["Eyes-Defy\nViT-B/16"], color="k", ls="--", lw=1, label="published 98.47%")
    ax.set_xlabel("epoch")
    ax.set_ylabel("test accuracy")
    ax.set_title("ViT test accuracy after each epoch (Eyes-Defy-Anemia)")
    ax.legend(fontsize=8, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "vit_epochs.png", dpi=150)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    replication_bars()
    epoch_curves()
    print(f"wrote {sorted(p.name for p in OUT.glob('*.png'))}")


if __name__ == "__main__":
    main()
