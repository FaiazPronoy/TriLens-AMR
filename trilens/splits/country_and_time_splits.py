from __future__ import annotations

import json
import pickle

import numpy as np
from sklearn.model_selection import train_test_split

from trilens.data.strain_metadata import AB_NAMES, PROCESSED, ROOT, load_strain_table

SEED = 42
VAL_FRAC = 0.15
CUTOFF_YEAR = 2013
SPLITS = {
    "country_uk_to_norway": dict(kind="country", train="United Kingdom", test="Norway"),
    "country_norway_to_uk": dict(kind="country", train="Norway", test="United Kingdom"),
    "time_norway_2013": dict(kind="time", country="Norway", cutoff=CUTOFF_YEAR),
}

ANTIBIOTICS = {
    "country_uk_to_norway": dict(primary=["AMP", "CIP", "GEN", "CTX", "CAZ"], secondary=[]),
    "country_norway_to_uk": dict(primary=["AMP", "CIP", "GEN", "CTX", "CAZ"], secondary=[]),
    "time_norway_2013": dict(primary=["AMP", "CIP", "SXT", "GEN"], secondary=["CTX", "CAZ"]),
}


def _sides(t, rule):
    if rule["kind"] == "country":
        return (t.country == rule["train"]).values, (t.country == rule["test"]).values
    inside = (t.country == rule["country"]).values & t.year.notna().values
    year = t.year.fillna(0).values
    return inside & (year <= rule["cutoff"]), inside & (year > rule["cutoff"])


def build_split(t, name):
    rule = SPLITS[name]
    train_side, test_side = _sides(t, rule)
    side_idx = np.flatnonzero(train_side)
    strata = t["AMP"].iloc[side_idx].fillna(-1).astype(int).values
    tr, va = train_test_split(side_idx, test_size=VAL_FRAC, random_state=SEED, stratify=strata)
    te = np.flatnonzero(test_side)
    used = np.zeros(len(t), bool)
    used[side_idx] = True
    used[te] = True
    split = {
        "train_idx": np.sort(tr).astype(np.int64),
        "val_idx": np.sort(va).astype(np.int64),
        "test_idx": te.astype(np.int64),
        "unused_idx": np.flatnonzero(~used).astype(np.int64),
        "st_per_genome": np.array(t.st.values, dtype=object),
        "rules": dict(
            rule, seed=SEED, val_frac=VAL_FRAC, val_stratified_on="AMP label, 'not tested' as its own stratum"
        ),
        "antibiotics": ANTIBIOTICS[name],
        "info": {},
    }
    for part in ("train", "val", "test"):
        idx = split[f"{part}_idx"]
        split["info"][part] = {
            "n": int(len(idx)),
            **{
                ab: dict(n_tested=int(t[ab].iloc[idx].notna().sum()), n_resistant=int((t[ab].iloc[idx] == 1).sum()))
                for ab in AB_NAMES
            },
        }
    return split


def check_split(t, name, split):
    tr, va, te = (set(split[k].tolist()) for k in ("train_idx", "val_idx", "test_idx"))
    assert not (tr & va) and not (tr & te) and not (va & te), "partitions overlap"
    assert len(tr | va | te) + len(split["unused_idx"]) == len(t), "partitions + unused != all strains"
    rule = SPLITS[name]
    fit = sorted(tr | va)
    if rule["kind"] == "country":
        assert (t.country.iloc[sorted(te)] == rule["test"]).all()
        assert (t.country.iloc[fit] == rule["train"]).all()
    else:
        assert (t.country.iloc[sorted(te)] == rule["country"]).all()
        assert (t.year.iloc[sorted(te)] > rule["cutoff"]).all()
        assert (t.country.iloc[fit] == rule["country"]).all()
        assert (t.year.iloc[fit] <= rule["cutoff"]).all()


def main():
    t = load_strain_table()
    out_json = ROOT / "results" / "splits"
    out_json.mkdir(parents=True, exist_ok=True)
    for name in SPLITS:
        split = build_split(t, name)
        check_split(t, name, split)
        with open(PROCESSED / f"{name}_split.pkl", "wb") as f:
            pickle.dump(split, f, protocol=4)
        (out_json / f"{name}.json").write_text(
            json.dumps(
                {
                    "name": name,
                    "rules": split["rules"],
                    "antibiotics": split["antibiotics"],
                    "info": split["info"],
                    **{k: split[k].tolist() for k in ("train_idx", "val_idx", "test_idx", "unused_idx")},
                }
            )
        )
        i = split["info"]
        print(
            f"{name:24s} train {i['train']['n']:5d}  val {i['val']['n']:4d}  test {i['test']['n']:5d}"
            f"  unused {len(split['unused_idx'])}"
        )


if __name__ == "__main__":
    main()
