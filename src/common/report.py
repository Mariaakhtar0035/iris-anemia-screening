"""Small helpers shared by the report writers: metric tables, the commit stamp, console section headers."""
import json


def fmt(m):
    return f"{m['estimate']:.3f} [{m['lo']:.3f}, {m['hi']:.3f}]"


IMAGE_COLUMNS = {"auroc": "AUROC", "pr_auc": "PR-AUC", "sens_at_90_spec": "Sens @ 90% spec"}


def markdown_table(rows, columns, metrics=IMAGE_COLUMNS):
    """rows: dicts holding the label columns plus one metric dict per key of `metrics`; a missing metric prints '-'."""
    head = columns + list(metrics.values())
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for r in rows:
        cells = [str(r[c]) for c in columns] + [fmt(r[m]) if m in r else "-" for m in metrics]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def short_commit(path):
    """Commit stamp read from a saved results JSON: 7-character hash, plus a note if the tree had uncommitted changes."""
    git = json.loads(path.read_text())["git"]
    return git["commit"][:7] + (" + uncommitted changes" if git["dirty"] else "")


def section(title):
    print(f"\n=== {title} ===")
