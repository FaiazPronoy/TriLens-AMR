"""Shuffle test: each run is cached, EXP3_ONLY runs a subset, the summary reuses the caches."""
import json
import types

import numpy as np
import pytest
from omegaconf import OmegaConf

pytest.importorskip("lightning")
import src.experiments.run_pillar1_exp3 as exp3  # noqa: E402

CALLS = []


def fake_run(dataset):
    def _run(ab, perm, order_tag, cfg, output_dir):
        CALLS.append(f"{dataset}_{ab}_{order_tag}")
        r = types.SimpleNamespace(
            y_true=np.array([0, 1]), y_prob=np.array([0.2, 0.8]),
            metrics={"pr_auc": 0.9 if order_tag == "orig" else 0.8, "mcc": 0.5},
            pr_auc_ci=(0.85, 0.95), mcc_ci=(0.4, 0.6), best_ckpt="x")
        return {"result": r, "n_snps": 10}
    return _run


@pytest.fixture
def cfg(tmp_path, monkeypatch):
    monkeypatch.setattr(exp3, "_run_arcadia", fake_run("arcadia"))
    monkeypatch.setattr(exp3, "_run_giessen", fake_run("giessen"))
    monkeypatch.setattr(exp3, "result_to_row", lambda d, ab, m, o, res, el: {
        "dataset": d, "antibiotic": ab, "variant": o, "status": "ok",
        "test_pr_auc": res.metrics["pr_auc"]})
    CALLS.clear()
    return OmegaConf.create({"seed": 1, "experiment": {
        "output_dir": str(tmp_path), "antibiotics": ["CIP"], "datasets": ["arcadia", "giessen"],
        "shuffle_seed": 1}})


def test_subset_then_summary_reuses_cached_runs(cfg, tmp_path, monkeypatch):
    monkeypatch.setenv("EXP3_ONLY", "giessen_CIP_orig")
    exp3.run(cfg)
    assert CALLS == ["giessen_CIP_orig"]
    assert (tmp_path / "giessen_CIP_orig_run.json").exists()
    assert not (tmp_path / "exp3_results.csv").exists()

    monkeypatch.delenv("EXP3_ONLY")
    exp3.run(cfg)
    assert sorted(CALLS) == sorted(["giessen_CIP_orig", "arcadia_CIP_orig", "arcadia_CIP_shuf", "giessen_CIP_shuf"])
    assert CALLS.count("giessen_CIP_orig") == 1                    # not trained twice
    import pandas as pd
    df = pd.read_csv(tmp_path / "exp3_results.csv")
    assert len(df) == 4 and set(df.status) == {"ok"}
    saved = json.loads((tmp_path / "exp3_results.json").read_text())
    assert saved["giessen_CIP_orig"]["metrics"]["pr_auc"] == 0.9
