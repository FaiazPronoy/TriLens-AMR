"""W6 runner on synthetic data: a gene only in the ARG block drives the primary drugs, a catalogue
locus drives the control. F0A must beat F0 on the primary drugs and not move the control."""
import numpy as np
import pandas as pd
import pytest

pytest.importorskip("xgboost")
import src.experiments.run_arg_rescue as rr  # noqa: E402


@pytest.fixture
def data(monkeypatch):
    monkeypatch.setitem(rr.XGB, "n_estimators", 40)
    rng = np.random.default_rng(0)
    n = 600
    st = np.array([str(i % 40) if i % 17 else None for i in range(n)], dtype=object)
    pa = rng.integers(0, 2, (n, 30)).astype(np.uint8)
    esbl = rng.integers(0, 2, n)
    arg = pd.DataFrame({"blaCTX-M-15": esbl, "noise": rng.integers(0, 2, n)}, dtype="uint8")
    flip = lambda p: (rng.random(n) < p).astype(float)          # noqa: E731
    Y = pd.DataFrame({
        "CTX": np.where(esbl == 1, 1 - flip(0.05), flip(0.05)),
        "CAZ": np.where(esbl == 1, 1 - flip(0.1), flip(0.05)),
        "GEN": np.where(esbl == 1, 1 - flip(0.1), flip(0.1)),
        "CIP": np.where(pa[:, 0] == 1, 1 - flip(0.05), flip(0.05)),
    })
    Y.iloc[::23, 0] = np.nan
    meta = pd.DataFrame({c: rng.random(n) for c in rr.META_NUM})
    for c in rr.META_CAT:
        meta[c] = rng.integers(0, 3, n)
    keep = np.ones(n, bool)
    keep[::50] = False
    return dict(Y=Y, pa=pa, arg=arg, meta=meta, keep=keep, st=st)


def folds_by_st(st, k=5):
    from src.splits.kfold import st_grouped
    fs, _ = st_grouped(st, k)
    n = len(st)
    return [(np.setdiff1d(np.arange(n), f), f) for f in fs]


def test_rescue_detects_missing_determinant(data, tmp_path):
    splits = {"st_grouped_5fold": folds_by_st(data["st"])}
    m, w, dl, v = rr.run(data, splits, ["F0", "F0A", "A", "M"], ["CTX", "CAZ", "GEN", "CIP"], tmp_path, n_boot=100)
    assert v["primary_pass"] and v["control_pass"]
    assert "supported" in v["reading"]
    S = pd.read_csv(tmp_path / "scores_st_grouped_5fold.csv.gz", index_col=0)
    assert S.loc[~data["keep"]].isna().all().all()                 # dropped strains never scored
    assert S.loc[[i for i, s in enumerate(data["st"]) if s is None]].isna().all().all()  # unknown ST never scored
    assert set(w.columns) >= {"macro", "n_st"}


def test_rescue_fails_when_block_is_uninformative(data, tmp_path):
    data["arg"]["blaCTX-M-15"] = np.random.default_rng(5).integers(0, 2, len(data["st"])).astype("uint8")
    splits = {"st_grouped_5fold": folds_by_st(data["st"])}
    *_, v = rr.run(data, splits, ["F0", "F0A"], ["CTX", "CAZ", "GEN", "CIP"], tmp_path, n_boot=100)
    assert not v["primary_pass"] and "not supported" in v["reading"]


def test_resume_skips_cached_columns(data, tmp_path, monkeypatch):
    splits = {"st_grouped_5fold": folds_by_st(data["st"])}
    rr.run(data, splits, ["F0"], ["CTX"], tmp_path, n_boot=10)
    monkeypatch.setattr(rr, "oof", lambda *a, **k: (_ for _ in ()).throw(AssertionError("refit")))
    rr.run(data, splits, ["F0"], ["CTX"], tmp_path, n_boot=10)


def test_metadata_arm_survives_an_empty_field():
    import pandas as pd
    rng = np.random.default_rng(0)
    X = pd.DataFrame({c: rng.random(200) for c in rr.META_NUM})
    X["Collection.Year"] = np.nan                       # no year in this fold's training strains
    for c in rr.META_CAT:
        X[c] = rng.integers(0, 3, 200)
    y = rng.integers(0, 2, 200)
    p = rr.fit_predict("M", X.iloc[:150], y[:150], X.iloc[150:])
    assert p.shape == (50,) and np.isfinite(p).all()
