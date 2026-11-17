"""Stage 29: feature selection from scratch on the Zimbabwe 2015 DHS (no transfer).

Steps 1-2 (this file so far): inventory every column of the Individual Recode among women with an Hb result, remove
what can't be a feature with a written reason (blood test, bookkeeping, repeated per-child copies, text, unit codes,
constant, mostly empty), and lock away 20% of villages as the final test. Two targets are saved: anemic yes/no (v457),
the main one, and the adjusted Hb number (v456) as an extra. Nothing here looks at how a column relates
to anemia, so none of these choices can leak the answer.

Run from the repo root: python -m src.survey.select_features
"""
import re

import numpy as np
import pandas as pd
import pyreadstat
from sklearn.model_selection import StratifiedGroupKFold

from src.common.io import ROOT, load_config, save_results
from src.eyelid.train_color import _short_commit

OUT = ROOT / "data" / "processed" / "stage29_zimbabwe.parquet"
REPEAT = re.compile(r"^(.+)_(\d+)$")


def read_tested(path, cfg):
    """Women with an Hb result, every column. Numbers stored as text (blank-and-integer columns) become numbers;
    columns that are real text (ids, the calendar) are returned separately."""
    raw, meta = pyreadstat.read_dta(path)
    raw = raw[raw[cfg["label"]].notna()].reset_index(drop=True)
    text = []
    for c in [c for c in raw.columns if not pd.api.types.is_numeric_dtype(raw[c])]:
        as_num = pd.to_numeric(raw[c], errors="coerce")
        if (as_num.isna() & raw[c].notna()).any():
            text.append(c)
        else:
            raw[c] = as_num
    return raw, meta, text


def labelled_share(values, value_labels):
    """Share of non-blank values that carry a value label."""
    v = values.dropna()
    if not len(v) or not value_labels:
        return 0.0
    return float(v.astype(float).isin({float(k) for k in value_labels}).mean())


def inventory(df, meta, text, cfg):
    """One row per column: label, share empty, distinct values, kind, and keep/drop with the first reason that applies."""
    labels, vlabels = meta.column_names_to_labels, meta.variable_value_labels
    leak = re.compile("|".join(cfg["leakage_label_words"]), re.IGNORECASE)
    admin = re.compile("|".join(cfg["admin_label_patterns"]), re.IGNORECASE)
    unit_words = [w.lower() for w in cfg["unit_coded_label_words"]]
    rows = []
    for c in df.columns:
        label = labels.get(c) or ""
        vl = vlabels.get(c, {})
        empty, n_distinct = float(df[c].isna().mean()), int(df[c].nunique(dropna=True))
        rep = REPEAT.match(c)
        if c in text:
            kind, n_distinct = "text", int(df[c].nunique(dropna=True))
        elif labelled_share(df[c], vl) >= cfg["category_min_labelled_share"]:
            kind = "category"
        else:
            kind = "number"
        if c == cfg["label"]:
            reason = "the anemia label"
        elif leak.search(label):
            reason = "leakage: blood test"
        elif label.lower().startswith(cfg["not_asked_label_prefix"]):
            reason = "not asked in this survey"
        elif admin.search(label):
            reason = "bookkeeping"
        elif rep and int(rep.group(2)) != cfg["keep_repeat_index"]:
            reason = "repeat: older birth or child"
        elif kind == "text":
            reason = "text"
        elif kind == "number" and any(w in str(v).lower() for v in vl.values() for w in unit_words):
            reason = "unit-coded time"
        elif n_distinct <= 1:
            reason = "constant"
        elif empty > cfg["max_share_empty"]:
            reason = "mostly empty"
        else:
            reason = ""
        rows.append({"code": c, "label": label, "share_empty": round(empty, 4), "n_distinct": n_distinct, "kind": kind,
                     "decision": "drop" if reason else "keep", "reason": reason})
    return pd.DataFrame(rows)


def feature_table(df, inv, meta):
    """Kept columns ready for models. Categories become their label text. Numbers stay numbers; their labelled codes
    ("don't know", "on premises", "flagged") are blanked and kept as text in a side column <code>_special."""
    vlabels, out = meta.variable_value_labels, {}
    for _, r in inv[inv.decision == "keep"].iterrows():
        c, vl = r.code, {float(k): v for k, v in meta.variable_value_labels.get(r.code, {}).items()}
        x = df[c].astype(float)
        if r.kind == "category":
            out[c] = x.map(lambda v: vl.get(v, str(int(v)) if float(v).is_integer() else str(v)) if pd.notna(v) else None)
        else:
            special = x.isin(set(vl))
            out[c] = x.where(~special)
            if special.any():
                out[f"{c}_special"] = np.where(special, x.map(vl), np.where(x.notna(), "value", None))
    return pd.DataFrame(out)


def lock_test(y, groups, n_splits, seed):
    """True for the women in the final-test villages: the first fold of a StratifiedGroupKFold by village."""
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    _, test = next(cv.split(np.zeros(len(y)), y, groups))
    mask = np.zeros(len(y), bool)
    mask[test] = True
    return mask


def main():
    base, cfg, survey = load_config("default"), load_config("feature_selection"), load_config("survey")
    seed = base["seed"]
    df, meta, text = read_tested(ROOT / survey["countries"][cfg["country"]], cfg)
    y = df[cfg["label"]].astype(int).isin(cfg["anemic_levels"]).astype(int).to_numpy()
    groups = df[cfg["cluster"]].astype(int).to_numpy()
    h = cfg["hb_target"]
    hb = (df[h["column"]] / h["divisor"]).to_numpy()
    cutoff = np.where(df[h["pregnant_column"]] == 1, h["cutoff_pregnant_g_dl"], h["cutoff_g_dl"])
    assert ((hb < cutoff).astype(int) == y).all(), "the Hb cutoffs don't reproduce v457"
    print(f"{cfg['country']}: {len(df)} women with an Hb result, {df.shape[1]} columns, {y.mean():.1%} anemic", flush=True)

    inv = inventory(df, meta, text, cfg)
    feats = feature_table(df, inv, meta)
    locked = lock_test(y, groups, cfg["locked_test"]["n_splits"], seed)
    assert not set(groups[locked]) & set(groups[~locked]), "a village is on both sides of the locked split"

    OUT.parent.mkdir(parents=True, exist_ok=True)
    targets = {"anemic": y, "hb": hb, "hb_cutoff": cutoff, "cluster": groups, "locked_test": locked}
    assert not set(targets) & set(feats.columns)
    feats.assign(**targets).to_parquet(OUT, index=False)
    inv.to_csv(ROOT / "reports" / "stage29_inventory.csv", index=False)

    usable = inv.reason.isin(["", "mostly empty"])  # what the emptiness rule acts on
    by_threshold = {t: int((usable & (inv.share_empty <= t)).sum()) for t in cfg["report_share_empty"]}
    reasons = inv[inv.decision == "drop"].reason.value_counts()
    kept = inv[inv.decision == "keep"]
    split = {"train": {"women": int((~locked).sum()), "villages": len(set(groups[~locked])),
                       "anemic": float(y[~locked].mean())},
             "locked test": {"women": int(locked.sum()), "villages": len(set(groups[locked])),
                             "anemic": float(y[locked].mean())}}
    path = save_results("stage29_inventory", {"n_women": len(df), "n_columns": int(df.shape[1]),
                                              "dropped_by_reason": reasons.to_dict(), "kept": int(len(kept)),
                                              "kept_by_kind": kept.kind.value_counts().to_dict(),
                                              "model_columns": int(feats.shape[1]), "kept_at_max_empty": by_threshold,
                                              "split": split},
                        {"default": base, "feature_selection": cfg, "survey": survey})

    def listing(reason):
        sub = inv[inv.reason == reason]
        return ", ".join(f"`{r.code}` {r.label}" for r in sub.itertuples())

    lines = [
        "# Stage 29: feature selection from scratch (Zimbabwe 2015)",
        "",
        f"Generated by `python -m src.survey.select_features` at commit `{_short_commit(path)}`; settings in "
        "`configs/feature_selection.yaml`. Zimbabwe only, no transfer.",
        "",
        f"- **Women:** {len(df)} aged 15-49 with an Hb result, {y.mean():.1%} anemic.",
        "- **Anemic = DHS `v457` at any severity:** WHO cutoffs for women on Hb adjusted for altitude and smoking, "
        "< 12.0 g/dL, or < 11.0 g/dL when pregnant.",
        f"- **Extra target, the Hb number:** `v456` (adjusted Hb, mean {hb.mean():.1f} g/dL, range {hb.min():.1f}-"
        f"{hb.max():.1f}). With the same cutoffs it reproduces `v457` for all {len(df)} women. It is a target only, "
        "never an input; a predicted Hb gets the woman's own cutoff afterwards.",
        "- Steps 1-2 never look at how a column relates to anemia, so nothing here can leak the answer.",
        "",
        "## Locked final test (Step 0)",
        "",
        "20% of villages are set aside now and not used again until the final score.",
        "",
        "| part | women | villages | anemic |",
        "|---|---|---|---|",
        *[f"| {k} | {v['women']} | {v['villages']} | {v['anemic']:.1%} |" for k, v in split.items()],
        "",
        "## Steps 1-2: from every column to the usable ones",
        "",
        f"The file has **{df.shape[1]} columns**. Each was kept or dropped for the first reason that applies, in this "
        "order. The full list with labels is in `reports/stage29_inventory.csv`.",
        "",
        "| reason | columns | what it means |",
        "|---|---|---|",
    ]
    meaning = {
        "the anemia label": "the answer itself",
        "leakage: blood test": "Hb, anemia, consent or referral for the test, hers or her children's: copying the answer",
        "not asked in this survey": "DHS labels these \"na - \": the question wasn't in Zimbabwe's questionnaire",
        "bookkeeping": "ids, sampling design, interviewer, dates, interview process, data-quality flags",
        "repeat: older birth or child": "the same question for older births (`_2`, `_02`...); the most recent is kept",
        "text": "free text such as the month-by-month calendar",
        "unit-coded time": "a time written as unit digit + number (1xx days, 3xx months), meaningless as a number",
        "constant": "every tested woman has the same answer",
        "mostly empty": f"more than {cfg['max_share_empty']:.0%} blank among tested women (asked of few women)",
    }
    for reason, text_ in meaning.items():
        lines.append(f"| {reason} | {int(reasons.get(reason, 0))} | {text_} |")
    lines += [
        f"| **kept** | **{len(kept)}** | {kept.kind.value_counts().get('category', 0)} categories, "
        f"{kept.kind.value_counts().get('number', 0)} numbers |",
        "",
        f"Numbers that also hold labelled codes (\"don't know\", \"on premises\") get a side column `<code>_special`, so "
        f"the models see **{feats.shape[1]} columns** in all.",
        "",
        "**How the emptiness cutoff changes the count** (columns that pass every other rule):",
        "",
        "| at most this share blank | columns kept |",
        "|---|---|",
        *[f"| {t:.0%} | {n} |" for t, n in by_threshold.items()],
        "",
        "## Removed for leakage",
        "",
        listing("leakage: blood test"),
        "",
        "## Removed as bookkeeping",
        "",
        listing("bookkeeping"),
        "",
        "## Removed as unit-coded time",
        "",
        listing("unit-coded time"),
        "",
        "## Kept",
        "",
        "| code | label | share blank | kind |",
        "|---|---|---|---|",
        *[f"| `{r.code}` | {r.label} | {r.share_empty:.1%} | {r.kind} |" for r in kept.itertuples()],
        "",
    ]
    text_out = "\n".join(lines)
    (ROOT / "reports" / "stage29_feature_selection.md").write_text(text_out)
    print(f"kept {len(kept)} of {df.shape[1]} columns -> {feats.shape[1]} model columns; dropped: {reasons.to_dict()}")
    print(f"split: {split}")


if __name__ == "__main__":
    main()
