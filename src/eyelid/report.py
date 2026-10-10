"""Shared formatting for results tables."""


def fmt(m):
    return f"{m['estimate']:.3f} [{m['lo']:.3f}, {m['hi']:.3f}]"


def markdown_table(rows, columns):
    """rows: dicts holding the label columns plus auroc / pr_auc / sens_at_90_spec metric dicts."""
    head = columns + ["AUROC", "PR-AUC", "Sens @ 90% spec"]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for r in rows:
        cells = [str(r[c]) for c in columns] + [fmt(r["auroc"]), fmt(r["pr_auc"]), fmt(r["sens_at_90_spec"])]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)
