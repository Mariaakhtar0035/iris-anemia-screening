"""Audit CP-AnemiC and write data/processed/cp_anemic_labels.csv.

Run from the repo root: python -m src.eyelid.audit
"""
import hashlib

import numpy as np
import pandas as pd
from PIL import Image

from src.common.io import ROOT, load_config, save_results

RAW = ROOT / "data" / "raw" / "cp_anemic"
OUT = ROOT / "data" / "processed" / "cp_anemic_labels.csv"
DUPS_OUT = ROOT / "data" / "processed" / "cp_anemic_duplicates.csv"
NEAR_OUT = ROOT / "data" / "processed" / "cp_anemic_near_duplicates.csv"
FOLDER_IS_ANEMIC = {"Anemic": True, "Non-anemic": False}


def section(title):
    print(f"\n=== {title} ===")


def dhash(path, n=16):
    """Difference hash of the image composited on mid-grey: n*n bits that survive re-saving and small crops."""
    im = Image.open(path).convert("RGBA")
    flat = Image.alpha_composite(Image.new("RGBA", im.size, (128, 128, 128, 255)), im).convert("L")
    g = np.asarray(flat.resize((n + 1, n)), dtype=float)
    return (g[:, 1:] > g[:, :-1]).flatten()


def photo_groups(df, max_bits, max_size_diff):
    """Link byte-level groups whose images look like the same photo; returns photo_group per row and the links."""
    reps = df.drop_duplicates("image_group").reset_index(drop=True)
    hashes = np.array([dhash(p) for p in reps.path])
    sizes = np.array([Image.open(p).size for p in reps.path], dtype=float)
    bits = (hashes[:, None, :] != hashes[None, :, :]).sum(-1)
    rel_size = (np.abs(sizes[:, None, :] - sizes[None, :, :]) / np.maximum(sizes[:, None, :], sizes[None, :, :])).max(-1)
    linked = np.triu((bits <= max_bits) & (rel_size <= max_size_diff), 1)

    parent = list(range(len(reps)))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    links = []
    for i, j in np.argwhere(linked):
        parent[find(i)] = find(j)
        links.append((reps.image_group[i], reps.image_group[j], int(bits[i, j])))
    rep_ids = reps.image_group.to_numpy()
    root_of = {rep_ids[i]: find(i) for i in range(len(reps))}
    members = pd.Series(rep_ids).groupby([root_of[g] for g in rep_ids]).transform("min")
    to_photo = dict(zip(rep_ids, members))
    return df.image_group.map(to_photo), pd.DataFrame(links, columns=["image_group_a", "image_group_b", "dhash_bits"])


def rate_table(df, by):
    t = df.groupby(by).anemic.agg(n="size", anemic="sum")
    t["anemic_pct"] = (100 * t.anemic / t.n).round(1)
    return t.sort_values("n", ascending=False).to_string()


def main():
    cutoff = load_config()["anemia_cutoff_g_dl"]["child_6_59_months"]
    df = pd.read_excel(RAW / "Anemia_Data_Collection_Sheet.xlsx")
    assert df.notna().all().all(), "the sheet has missing values"
    assert set(df.REMARK) == {"Anemic", "Non-anemic"}, f"unexpected REMARK values: {set(df.REMARK)}"

    # The folder an image sits in is a third label source, next to Hb and REMARK.
    on_disk = {p.stem: p for folder in FOLDER_IS_ANEMIC for p in (RAW / folder).glob("*.png")}
    missing = sorted(set(df.IMAGE_ID) - set(on_disk))
    assert not missing, f"rows with no image file: {missing}"

    df["path"] = df.IMAGE_ID.map(on_disk)
    df["anemic"] = df.HB_LEVEL < cutoff
    df["remark_anemic"] = df.REMARK.eq("Anemic")
    df["folder_anemic"] = df.path.map(lambda p: FOLDER_IS_ANEMIC[p.parent.name])

    section("Counts")
    print(f"rows: {len(df)}, unique IMAGE_IDs: {df.IMAGE_ID.nunique()}, image files: {len(on_disk)}")
    print(f"image files with no row: {sorted(set(on_disk) - set(df.IMAGE_ID)) or 'none'}")
    print("patients: unknown. The sheet has no patient ID, so we assume one child per image.")

    section("Duplicates")
    dup_ids = df.IMAGE_ID[df.IMAGE_ID.duplicated()].tolist()
    print(f"duplicate IMAGE_IDs: {dup_ids or 'none'}")
    # Byte-identical files under different IDs. Copies must share a fold later, or a model is tested on images it trained on.
    df["sha256"] = df.path.map(lambda p: hashlib.sha256(p.read_bytes()).hexdigest())
    df["image_group"] = df.groupby("sha256").IMAGE_ID.transform("first")
    sizes = df.groupby("image_group").size()
    groups = sizes[sizes > 1]
    print(f"unique image files: {len(sizes)} of {len(df)} rows")
    print(f"groups of byte-identical files: {len(groups)}, rows in them: {groups.sum()}, extra copies: {groups.sum() - len(groups)}")
    print(f"group sizes: {groups.value_counts().sort_index().to_dict()}")
    in_groups = df[df.image_group.isin(groups.index)]
    differ = in_groups.groupby("image_group")[["HB_LEVEL", "anemic", "HOSPITAL", "Age(Months)", "GENDER"]].nunique().gt(1).sum()
    print(f"groups whose copies differ on a field (of {len(groups)}): {differ.to_dict()}")
    conflict = in_groups.groupby("image_group").anemic.transform("nunique").gt(1)
    if conflict.any():
        print("groups whose copies disagree on the anemia label:")
        print(in_groups[conflict][["image_group", "IMAGE_ID", "HB_LEVEL", "HOSPITAL"]].to_string(index=False))
    DUPS_OUT.parent.mkdir(parents=True, exist_ok=True)
    in_groups.sort_values(["image_group", "IMAGE_ID"])[
        ["image_group", "IMAGE_ID", "HB_LEVEL", "anemic", "HOSPITAL", "Age(Months)", "GENDER"]
    ].to_csv(DUPS_OUT, index=False)
    print(f"full list: {DUPS_OUT.relative_to(ROOT)}")

    section("Near-duplicates (same photo re-saved or slightly re-cropped)")
    nd = load_config()["cp_anemic_near_duplicate"]
    df["photo_group"], links = photo_groups(df, nd["dhash_max_bits"], nd["max_size_diff"])
    photos = df.photo_group.nunique()
    print(f"rule: dHash within {nd['dhash_max_bits']} of 256 bits and width/height within {nd['max_size_diff']:.0%}")
    print(f"links between byte-distinct images: {len(links)}; distinct photos: {photos} (byte-distinct images: {df.image_group.nunique()})")
    label_mixed = df.groupby("photo_group").anemic.nunique().gt(1)
    print(f"photos whose rows disagree on the anemia label: {int(label_mixed.sum())} -> {sorted(label_mixed[label_mixed].index)}")
    links.to_csv(NEAR_OUT, index=False)
    print(f"links: {NEAR_OUT.relative_to(ROOT)}")

    section(f"Anemia split (Hb < {cutoff} g/dL)")
    n_anemic = int(df.anemic.sum())
    print(f"anemic: {n_anemic} ({100 * n_anemic / len(df):.1f}%), not anemic: {len(df) - n_anemic}")
    print(f"rows with Hb exactly {cutoff} (labelled not anemic): {int((df.HB_LEVEL == cutoff).sum())}")

    section("Agreement of the Hb label with REMARK and with the folder")
    for col in ["remark_anemic", "folder_anemic"]:
        print(f"\nHb label vs {col}:")
        print(pd.crosstab(df.anemic, df[col], rownames=["hb_anemic"]).to_string())
    disagree = df[(df.anemic != df.remark_anemic) | (df.anemic != df.folder_anemic)]
    print(f"\nrows where any of the three disagree: {len(disagree)}")
    if len(disagree):
        cols = ["IMAGE_ID", "HB_LEVEL", "REMARK", "Severity", "folder_anemic", "HOSPITAL"]
        print(disagree[cols].to_string(index=False))

    section("By hospital (site)")
    print(rate_table(df, "HOSPITAL"))
    section("By region")
    print(rate_table(df, "REGION"))

    section("Age (months)")
    print(df["Age(Months)"].describe().round(1).to_string())
    print(f"\nrows over 59 months (outside the 6-59 month definition): {int((df['Age(Months)'] > 59).sum())}")
    print(f"rows under 6 months: {int((df['Age(Months)'] < 6).sum())}")
    print("\nmost common ages:", df["Age(Months)"].value_counts().head(10).to_dict())

    section("Sex")
    print(rate_table(df, "GENDER"))

    section("Image sizes")
    sizes = df.path.map(lambda p: Image.open(p).size)
    modes = df.path.map(lambda p: Image.open(p).mode)
    w, h = sizes.str[0], sizes.str[1]
    print(f"width  min {w.min()}, median {int(w.median())}, max {w.max()}")
    print(f"height min {h.min()}, median {int(h.median())}, max {h.max()}")
    print(f"distinct sizes: {sizes.nunique()}; most common: {sizes.value_counts().head(5).to_dict()}")
    print(f"color modes: {modes.value_counts().to_dict()}")
    # A transparent background would mean the conjunctiva is already cut out, which matters for the Stage 3 region of interest.
    alpha_min = df.path.map(lambda p: Image.open(p).convert("RGBA").getchannel("A").getextrema()[0])
    print(f"images with some transparent pixels: {int((alpha_min < 255).sum())} of {len(df)}")

    per_photo = df.groupby("photo_group").size()
    facts = {
        "rows": len(df),
        "byte_distinct_images": int(df.image_group.nunique()),
        "byte_identical_groups": int(len(groups)),
        "extra_byte_identical_copies": int(groups.sum() - len(groups)),
        "near_duplicate_links": int(len(links)),
        "distinct_photos": int(photos),
        "photos_with_no_copy": int((per_photo == 1).sum()),
        "photos_with_conflicting_labels": int(label_mixed.sum()),
        "anemic_rows": n_anemic,
        "anemic_share_rows": round(n_anemic / len(df), 4),
        "hb_label_disagrees_with_remark_or_folder": int(len(disagree)),
        "rows_over_59_months": int((df["Age(Months)"] > 59).sum()),
        "hospitals": int(df.HOSPITAL.nunique()),
        "anemic_pct_by_hospital": (100 * df.groupby("HOSPITAL").anemic.mean()).round(1).to_dict(),
        "anemia_cutoff_g_dl": cutoff,
    }
    save_results("stage2_audit", facts, {"default": load_config()})

    OUT.parent.mkdir(parents=True, exist_ok=True)
    labels = pd.DataFrame({
        "id": df.IMAGE_ID,
        "image_path": df.path.map(lambda p: str(p.relative_to(ROOT))),
        "hb": df.HB_LEVEL,
        "anemic": df.anemic.astype(int),
        "site": df.HOSPITAL,
        "age_months": df["Age(Months)"],
        "sex": df.GENDER,
        "image_group": df.image_group,
        "photo_group": df.photo_group,
    })
    labels.to_csv(OUT, index=False)
    section("Output")
    print(f"wrote {OUT.relative_to(ROOT)}: {len(labels)} rows, columns {list(labels.columns)}")


if __name__ == "__main__":
    main()
