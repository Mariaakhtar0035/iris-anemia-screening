"""Data funnel: women left at each cleaning step, per country. Writes results/funnel.csv and reports/figures/funnel.png.

Steps: everyone in the Individual Recode, then women with an Hb result (what build_dataset keeps), then women with no
missing model feature. The models impute missing features instead of dropping them, so the last step only shows what a
complete-case analysis would lose.

Run from the repo root: python -m src.survey.funnel
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.common.io import RESULTS, ROOT, load_config
from src.survey.build_dataset import clean, read_raw

FIGURE = ROOT / "reports" / "figures" / "funnel.png"
STEPS = ["in file", "Hb result", "no missing feature"]
COLORS = ["#86b6ef", "#2a78d6", "#104281"]  # one blue, light to dark: later steps darker
INK, INK_2, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def funnel(country, cfg):
    raw, labels = read_raw(country, cfg)
    tested = clean(raw, labels, cfg, country)
    features = [f for group in cfg["features"].values() for f in group]
    missing = tested[features].isna()
    complete = tested[~missing.any(axis=1)]
    counts = [len(raw), len(tested), len(complete)]
    rows = [{"country": country, "step": s, "n": n, "pct_of_file": n / counts[0], "dropped": prev - n}
            for s, n, prev in zip(STEPS, counts, [counts[0]] + counts[:-1])]
    return rows, missing.sum()[missing.sum() > 0].to_dict()


def plot(df, path):
    countries = list(dict.fromkeys(df.country))
    width = 0.26
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for i, (step, color) in enumerate(zip(STEPS, COLORS)):
        sub = df[df.step == step].set_index("country").loc[countries]
        x = np.arange(len(countries)) + (i - 1) * (width + 0.02)
        ax.bar(x, sub.n, width, color=color, label=step, zorder=2)
        for xi, n, pct in zip(x, sub.n, sub.pct_of_file):
            ax.text(xi, n, f"{n:,}\n{pct:.0%}", ha="center", va="bottom", fontsize=8, color=INK_2)
    ax.set_xticks(np.arange(len(countries)), [c.capitalize() for c in countries], color=INK)
    ax.set_ylabel("women", color=INK_2)
    ax.yaxis.set_major_formatter(matplotlib.ticker.StrMethodFormatter("{x:,.0f}"))
    ax.set_ylim(0, df.n.max() * 1.15)
    ax.grid(axis="y", color=GRID, zorder=0)
    ax.tick_params(colors=INK_2, length=0)
    for side in ["top", "right", "left"]:
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(INK_2)
    ax.legend(frameon=False, loc="upper right", labelcolor=INK_2)
    ax.set_title("Women left at each cleaning step (% of everyone in the file)", color=INK, loc="left")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)


def main():
    cfg = load_config("survey")
    rows = []
    for country in cfg["countries"]:
        country_rows, missing = funnel(country, cfg)
        rows += country_rows
        print(f"{country}: " + " -> ".join(f"{r['n']} ({r['pct_of_file']:.1%})" for r in country_rows)
              + f"; missing features among tested: {missing or 'none'}")
    df = pd.DataFrame(rows)
    RESULTS.mkdir(exist_ok=True)
    df.to_csv(RESULTS / "funnel.csv", index=False)
    plot(df, FIGURE)
    print(f"-> {(RESULTS / 'funnel.csv').relative_to(ROOT)}, {FIGURE.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
