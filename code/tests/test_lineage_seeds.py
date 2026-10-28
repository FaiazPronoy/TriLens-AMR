"""W9 summary rule: pass needs all three seeds, mean delta > 0 and CI above zero in >= 2 seeds."""
import json

import pandas as pd

import src.experiments.run_lineage_seeds as L


def _write(root, seed, delta, lo):
    d = root / f"seed_{seed}" / "diagnostics"
    d.mkdir(parents=True)
    pd.DataFrame([dict(comparison="stack - classical", endpoint="mean_7", delta_nap=delta, ci_low=lo,
                       ci_high=delta + 0.02, n_st=100)]).to_csv(d / "holdout_validity_deltas.csv", index=False)
    pd.DataFrame([dict(model=m, drug="MEAN", ap=0.6, nap=0.5, auroc=0.8, within_st_macro=0.6)
                  for m in ("stack", "classical", "neural")]).to_csv(d / "holdout_validity.csv", index=False)


def test_rule(tmp_path, monkeypatch):
    monkeypatch.setattr(L, "OUT", tmp_path)
    _write(tmp_path, 42, 0.028, 0.006)
    _write(tmp_path, 123, 0.020, -0.002)
    assert not L.summary([42, 123, 456])["primary_pass"]          # only two seeds read
    _write(tmp_path, 456, 0.025, 0.003)
    v = L.summary([42, 123, 456])
    assert v["primary_pass"] and v["seeds_with_ci_above_zero"] == 2
    assert json.loads((tmp_path / "verdict.json").read_text())["complete"]


def test_fails_when_two_seeds_cross_zero(tmp_path, monkeypatch):
    monkeypatch.setattr(L, "OUT", tmp_path)
    for s, d, lo in [(42, 0.028, 0.006), (123, 0.01, -0.01), (456, 0.005, -0.02)]:
        _write(tmp_path, s, d, lo)
    assert not L.summary([42, 123, 456])["primary_pass"]
