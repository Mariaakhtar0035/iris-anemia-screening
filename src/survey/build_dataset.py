"""Build data/processed/{country}.parquet from each DHS Individual Recode: women with an Hb result, clean columns.

Run from the repo root: python -m src.survey.build_dataset
"""
import pandas as pd
import pyreadstat

from src.common.io import ROOT, load_config

OUT = ROOT / "data" / "processed"


def read_raw(country, cfg):
    """All women in the file, the configured variables renamed to plain names, plus the value labels."""
    v = cfg["variables"]
    df, meta = pyreadstat.read_dta(ROOT / cfg["countries"][country], usecols=list(v.values()))
    return df.rename(columns={code: name for name, code in v.items()}), meta.variable_value_labels


def clean(raw, labels, cfg, country):
    """Keep women with an Hb result; add the binary label and decoded categories."""
    v = cfg["variables"]
    df = raw[raw.anemia_level.notna()].copy()
    return pd.DataFrame({
        "country": country,
        "cluster": df.cluster.astype(int),
        "weight": df.weight / cfg["weight_divisor"],
        "age": df.age.astype(int),
        "pregnant": df.pregnant.astype(int),
        "wealth": df.wealth,  # kept numeric 1-5; missing stays missing and is imputed inside the model pipeline
        "education": df.education,
        "contraception": df.contraception.map(labels[v["contraception"]]),
        "residence": df.residence.map(labels[v["residence"]]),
        "region": df.region.map(labels[v["region"]]),
        "hb_adjusted": df.hb_adjusted / 10,
        "anemic": df.anemia_level.isin(cfg["anemic_levels"]).astype(int),
    }).reset_index(drop=True)


def load(country):
    return pd.read_parquet(OUT / f"{country}.parquet")


def main():
    cfg = load_config("survey")
    OUT.mkdir(parents=True, exist_ok=True)
    for country in cfg["countries"]:
        raw, labels = read_raw(country, cfg)
        df = clean(raw, labels, cfg, country)
        df.to_parquet(OUT / f"{country}.parquet", index=False)
        print(f"{country}: {len(df)} women with an Hb result (of {len(raw)}), {df.anemic.mean():.1%} anemic "
              f"-> {(OUT / f'{country}.parquet').relative_to(ROOT)}")


if __name__ == "__main__":
    main()
