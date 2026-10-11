"""Audit Eyes-Defy-Anemia and write data/processed/eyes_defy_labels.csv.

Run from the repo root: python -m src.eyelid.audit_eyes_defy
"""
import hashlib

import pandas as pd
from PIL import Image

from src.common.io import ROOT, load_config, save_results
from src.eyelid.audit import photo_groups, section

RAW = ROOT / "data" / "raw" / "eyes_defy"
OUT = ROOT / "data" / "processed" / "eyes_defy_labels.csv"
COUNTRIES = ("Italy", "India")
PAPER_COUNTS = {"photos": 217, "no_anemia": 126, "anemia": 91}  # Sci Rep 2025, "Dataset" section


def load_country(country):
    df = pd.read_excel(RAW / country / f"{country}.xlsx")
    folders = {int(d.name) for d in (RAW / country).iterdir() if d.name.isdigit()}
    assert folders == set(df.Number), f"{country}: folders and Number column differ"
    jpgs = {n: sorted((RAW / country / str(n)).glob("*.jpg")) for n in df.Number}
    assert all(len(v) == 1 for v in jpgs.values()), f"{country}: a folder without exactly one eye photo"
    # Italy writes some Hb values with a decimal comma ("15,1") and patient 93 as "_".
    hb = pd.to_numeric(df.Hgb.astype(str).str.replace(",", "."), errors="coerce")
    return pd.DataFrame({
        "id": [f"{country}_{n}" for n in df.Number],
        "country": country,
        "number": df.Number,
        "path": [jpgs[n][0] for n in df.Number],
        "hb": hb,
        "hb_raw": df.Hgb.astype(str),
        "sex": df.Gender,
        "age": df.Age,
        "note": df.Note,
    })


def paper_label(df, c):
    """The paper's label: India by sex (< 12 women, < 14 men), Italy one cutoff (<= 10.5, see the config)."""
    india = (df.country == "India") & (
        ((df.sex == "F") & (df.hb < c["india_female_below"])) | ((df.sex == "M") & (df.hb < c["india_male_below"])))
    italy = (df.country == "Italy") & (df.hb <= c["italy_at_or_below"])
    return (india | italy).astype(float).where(df.hb.notna())


def main():
    cfg = load_config("replicate_vit")
    cut = cfg["paper_cutoffs_g_dl"]
    df = pd.concat([load_country(c) for c in COUNTRIES], ignore_index=True)

    section("Counts")
    print(df.groupby("country").size().to_dict(), "photos")
    no_hb = df[df.hb.isna()]
    print(f"photos with no usable Hb: {len(no_hb)} -> {no_hb[['id', 'hb_raw', 'note']].to_dict('records')}")
    print(f"Hb values written with a decimal comma: {int(df.hb_raw.str.contains(',').sum())}")
    print("sex by country:", df.groupby(["country", "sex"]).size().to_dict())

    section("Duplicates")
    df["image_group"] = df.path.map(lambda p: hashlib.sha256(p.read_bytes()).hexdigest()[:16])
    print(f"byte-distinct photos: {df.image_group.nunique()} of {len(df)}")
    nd = load_config()["cp_anemic_near_duplicate"]
    df["photo_group"], links = photo_groups(df, nd["dhash_max_bits"], nd["max_size_diff"])
    print(f"same CP-AnemiC rule (dHash within {nd['dhash_max_bits']} bits, size within {nd['max_size_diff']:.0%}): "
          f"{len(links)} links, {df.photo_group.nunique()} distinct photos")
    if len(links):
        print(links.to_string(index=False))

    section(f"Paper labels ({cut})")
    df["anemic_paper"] = paper_label(df, cut)
    used = df[df.anemic_paper.notna()]
    n_an = int(used.anemic_paper.sum())
    print(pd.crosstab(used.country, used.anemic_paper.map({0.0: "no anemia", 1.0: "anemia"})).to_string())
    print(f"total: {len(used)} photos, {len(used) - n_an} no anemia / {n_an} anemia; paper: {PAPER_COUNTS}")
    at_cut = used[(used.country == "Italy") & (used.hb == cut["italy_at_or_below"])]
    print(f"Italy photos with Hb exactly at the cutoff (anemic only under <=): {at_cut.id.tolist()}")

    section("Image sizes")
    sizes = df.path.map(lambda p: Image.open(p).size)
    print("most common:", sizes.value_counts().head(5).to_dict())
    exif_rot = df.path.map(lambda p: Image.open(p).getexif().get(0x0112, 1))
    print(f"photos with an EXIF rotation tag other than 1: {int((exif_rot != 1).sum())}")

    facts = {
        "photos": len(df),
        "photos_by_country": df.groupby("country").size().to_dict(),
        "photos_without_hb": no_hb.id.tolist(),
        "byte_distinct_photos": int(df.image_group.nunique()),
        "near_duplicate_links": int(len(links)),
        "distinct_photos": int(df.photo_group.nunique()),
        "labelled_photos": len(used),
        "no_anemia": len(used) - n_an,
        "anemia": n_an,
        "paper_counts": PAPER_COUNTS,
        "matches_paper": (len(used), len(used) - n_an, n_an) == tuple(PAPER_COUNTS.values()),
        "italy_at_cutoff": at_cut.id.tolist(),
        "paper_cutoffs_g_dl": cut,
        "image_sizes": {f"{w}x{h}": int(n) for (w, h), n in sizes.value_counts().items()},
        "exif_rotated": int((exif_rot != 1).sum()),
    }
    save_results("stage_eyes_defy_audit", facts, {"replicate_vit": cfg, "default": load_config()})

    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.assign(image_path=df.path.map(lambda p: str(p.relative_to(ROOT))))[
        ["id", "country", "image_path", "hb", "sex", "age", "anemic_paper", "image_group", "photo_group"]
    ].to_csv(OUT, index=False)
    section("Output")
    print(f"wrote {OUT.relative_to(ROOT)}: {len(df)} rows")


if __name__ == "__main__":
    main()
