"""Audit the DHS Individual Recodes: variables, missing shares, anemia prevalence, and coding differences between countries.

Run from the repo root: python -m src.survey.audit
"""
import pyreadstat

from src.common.io import ROOT, load_config, save_results
from src.eyelid.audit import section
from src.survey.build_dataset import clean, read_raw


def main():
    cfg = load_config("survey")
    v = cfg["variables"]
    train = cfg["train_country"]
    facts, labels_by_country = {}, {}
    for country, path in cfg["countries"].items():
        _, meta = pyreadstat.read_dta(ROOT / path, metadataonly=True)
        var_labels = dict(zip(meta.column_names, meta.column_labels))
        missing_vars = [code for code in v.values() if code not in var_labels]
        assert not missing_vars, f"{country}: variables not in the file: {missing_vars}"

        raw, labels = read_raw(country, cfg)
        labels_by_country[country] = labels
        df = clean(raw, labels, cfg, country)
        section(f"{country} ({path})")
        for name, code in v.items():
            print(f"  {name:13s} {code:6s} {var_labels[code]}  (missing {raw[name].isna().mean():.1%})")
        n_no_hb = len(raw) - len(df)
        print(f"women: {len(raw)}; no Hb result: {n_no_hb} ({n_no_hb / len(raw):.1%}), dropped; kept {len(df)}")
        print(f"clusters with a tested woman: {df.cluster.nunique()} (of {raw.cluster.nunique()})")
        by_preg = df.groupby("pregnant").anemic.agg(["mean", "size"])
        print(f"anemic: {df.anemic.mean():.1%} overall; pregnant {by_preg.loc[1, 'mean']:.1%} (n={by_preg.loc[1, 'size']}), "
              f"not pregnant {by_preg.loc[0, 'mean']:.1%} (n={by_preg.loc[0, 'size']})")
        print("anemia level:", raw.anemia_level.map(labels[v["anemia_level"]]).value_counts().to_dict())
        wealth_gap = df[df.wealth.isna()]
        if len(wealth_gap):
            print(f"tested women with no wealth index: {len(wealth_gap)}, by region {wealth_gap.region.value_counts().to_dict()}")
        facts[country] = {
            "women": len(raw), "no_hb_result": n_no_hb, "tested": len(df), "clusters": int(df.cluster.nunique()),
            "anemic_share": round(df.anemic.mean(), 4),
            "anemic_share_pregnant": round(by_preg.loc[1, "mean"], 4),
            "anemic_share_not_pregnant": round(by_preg.loc[0, "mean"], 4),
            "missing_share_all_women": {k: round(raw[k].isna().mean(), 4) for k in v},
            "tested_missing_wealth": len(wealth_gap),
            "tested_missing_wealth_by_region": wealth_gap.region.value_counts().to_dict(),
            **{f"{c}_share": df[c].value_counts(normalize=True).round(4).to_dict() for c in cfg["features"]["categorical"]},
        }

    section(f"Codings compared with {train}")
    differences = {}
    for country, labels in labels_by_country.items():
        if country == train:
            continue
        for name, code in v.items():
            a, b = labels_by_country[train].get(code, {}), labels.get(code, {})
            if name == "region" or a == b:
                continue
            only_target = {k: b[k] for k in b if k not in a}
            changed = {k: (a[k], b[k]) for k in a if k in b and a[k] != b[k]}
            only_train = {k: a[k] for k in a if k not in b}
            differences.setdefault(country, {})[name] = {"only_here": only_target, "only_in_train": only_train,
                                                         "relabelled": changed}
            print(f"{country} {name} ({code}): new {only_target or '-'}; dropped {only_train or '-'}; relabelled {changed or '-'}")
    print("region (v024) differs by design: each country has its own regions, so it is not a model input.")
    # Labels can match while the codes actually used differ; a category the model never saw in training matters most.
    unseen = {}
    for col in cfg["features"]["categorical"]:
        seen = set(facts[train][f"{col}_share"])
        for country in facts:
            if country == train:
                continue
            new = {k: facts[country][f"{col}_share"][k] for k in facts[country][f"{col}_share"] if k not in seen}
            if new:
                unseen.setdefault(country, {})[col] = new
                print(f"{country} {col}: categories absent from {train}'s tested women (share of tested): {new}")

    save_results("stage6_survey_audit", {"countries": facts, "coding_differences": differences,
                                          "categories_unseen_in_train": unseen}, {"survey": cfg})


if __name__ == "__main__":
    main()
