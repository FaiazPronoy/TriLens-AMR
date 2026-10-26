import json
import pickle

import numpy as np
import pandas as pd
import pytest

from src.data.metadata import AB_NAMES, PROCESSED, ROOT
from src.splits import country_time as ct

HAVE_DATA = (ROOT / "data" / "raw" / "strains_metadata.csv").exists() and \
    (PROCESSED / "arcadia_v2_master_features.pkl").exists()


def _toy(n=400, seed=0):
    rng = np.random.default_rng(seed)
    t = pd.DataFrame({
        "country": rng.choice(["United Kingdom", "Norway", "USA", None], n, p=[.45, .4, .1, .05]),
        "year": [int(y) if y > 0 else None for y in rng.choice([-1, 2005, 2010, 2013, 2014, 2016], n)],
        "st": rng.choice(["131", "73", "95", None], n),
    })
    for ab in AB_NAMES:
        t[ab] = np.where(rng.random(n) < 0.8, rng.integers(0, 2, n), np.nan).astype(float)
    return t


@pytest.mark.parametrize("name", list(ct.SPLITS))
def test_toy_split_rules(name):
    t = _toy()
    split = ct.build_split(t, name)
    ct.check_split(t, name, split)
    again = ct.build_split(t, name)
    for k in ("train_idx", "val_idx", "test_idx", "unused_idx"):
        assert np.array_equal(split[k], again[k])


def test_check_split_catches_a_test_strain_in_train():
    t = _toy()
    split = ct.build_split(t, "country_uk_to_norway")
    split["train_idx"] = np.r_[split["train_idx"], split["test_idx"][:1]]
    with pytest.raises(AssertionError):
        ct.check_split(t, "country_uk_to_norway", split)


def test_time_split_keeps_late_years_out_of_training():
    t = _toy()
    split = ct.build_split(t, "time_norway_2013")
    fit = np.r_[split["train_idx"], split["val_idx"]]
    assert (t.year.iloc[fit] <= 2013).all() and (t.year.iloc[split["test_idx"]] > 2013).all()
    assert (t.country.iloc[np.r_[fit, split["test_idx"]]] == "Norway").all()


@pytest.mark.skipif(not HAVE_DATA, reason="small_data.zip not unpacked into code/data/")
@pytest.mark.parametrize("name", list(ct.SPLITS))
def test_real_split_files(name):
    from src.data.metadata import load_strain_table
    t = load_strain_table()
    built = ct.build_split(t, name)
    ct.check_split(t, name, built)
    p = PROCESSED / f"{name}_split.pkl"
    if p.exists():
        with open(p, "rb") as f:
            saved = pickle.load(f)
        for k in ("train_idx", "val_idx", "test_idx"):
            assert np.array_equal(saved[k], built[k]), f"{p.name}: {k} differs from a fresh build"
    j = ROOT / "results" / "splits" / f"{name}.json"
    if j.exists():
        d = json.loads(j.read_text())
        for k in ("train_idx", "val_idx", "test_idx"):
            assert d[k] == built[k].tolist()
    for ab in built["antibiotics"]["primary"]:
        te = built["info"]["test"][ab]
        assert te["n_resistant"] >= 50 and te["n_tested"] - te["n_resistant"] >= 50
