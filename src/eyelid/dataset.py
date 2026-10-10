"""CP-AnemiC subsets. The audit found far fewer distinct photos than rows (see README).

photo_group joins byte-identical copies and near-duplicates (same photo re-saved or re-cropped)."""
import pandas as pd

from src.common.io import ROOT

LABELS = ROOT / "data" / "processed" / "cp_anemic_labels.csv"


def load_cp_anemic(subset="unique"):
    """subset:
    all        every row, duplicates included (only for measuring leakage)
    unique     one row per distinct photo, minus photos whose copies disagree on the label
    singletons only photos with no copy, the only rows whose hospital we can trust
    """
    labels = pd.read_csv(LABELS)
    group = labels.groupby("photo_group")
    if subset == "all":
        return labels
    if subset == "unique":
        consistent = group.anemic.transform("nunique") == 1
        return labels[consistent].drop_duplicates("photo_group").reset_index(drop=True)
    if subset == "singletons":
        return labels[group.id.transform("size") == 1].reset_index(drop=True)
    raise ValueError(f"unknown subset {subset!r}")
