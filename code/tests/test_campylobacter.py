"""W13 Campylobacter: the rule, the 23S switch, and an end-to-end run on synthetic isolates."""
import importlib.util as iu
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("xgboost")
spec = iu.spec_from_file_location("cb", Path(__file__).resolve().parents[1] / "scripts/diagnostics/campylobacter.py")
cb = iu.module_from_spec(spec)
spec.loader.exec_module(cb)


@pytest.mark.parametrize("drug,acq,pt,with23s,expect", [
    ("CIP", set(), {"gyrA_T86I"}, True, 1.0), ("NAL", set(), {"gyrA_T86I"}, True, 1.0),
    ("CIP", {"gyrA_T86I"}, set(), True, 0.0),             # a gene named gyrA is not a point mutation
    ("TET", {"tet(O)"}, set(), True, 1.0), ("TET", {"blaOXA-61"}, set(), True, 0.0),
    ("CLI", set(), {"23S_A2075G"}, True, 1.0), ("CLI", set(), {"23S_A2075G"}, False, 0.0),
    ("CLI", {"erm(B)"}, set(), False, 1.0), ("CLI", {"lnu(C)"}, set(), False, 1.0),
    ("CLI", set(), {"rplV_A103V"}, True, 0.0),
])
def test_rules(drug, acq, pt, with23s, expect):
    assert cb.rule_score(acq, pt, cb.rules(with23s)[drug]) == expect


def _cohort(n, rng):
    rows = []
    for i in range(n):
        gy, tet, s23 = rng.random() < 0.25, rng.random() < 0.5, rng.random() < 0.05
        g = (["gyrA_T86I=POINT"] if gy else []) + (["tet(O)"] if tet else []) + \
            (["23S_A2075G=POINT"] if s23 else []) + ["blaOXA-61"] + (["tet(O)=PARTIAL"] if i % 7 == 0 else [])
        flip = lambda x: x if rng.random() > 0.05 else not x                    # noqa: E731
        ast = (f"ciprofloxacin={'R' if flip(gy) else 'S'},nalidixic acid={'R' if flip(gy) else 'S'},"
               f"tetracycline={'R' if flip(tet) else 'S'},clindamycin={'R' if flip(s23 or rng.random() < 0.02) else 'S'}")
        rows.append(dict(AMR_genotypes=",".join(g), AST_phenotypes=ast, PDS_acc=f"PDS{i % 80}" if i % 13 else None,
                         asm_stats_length_bp=str(1_700_000 + rng.integers(0, 50_000)),
                         asm_stats_n_contig=str(rng.integers(20, 80)), asm_stats_contig_n50=str(rng.integers(30_000, 200_000))))
    return pd.DataFrame(rows)


def test_end_to_end(tmp_path, monkeypatch):
    monkeypatch.setitem(cb.XGB, "n_estimators", 20)
    p = tmp_path / "c.tsv.gz"
    _cohort(1500, np.random.default_rng(0)).to_csv(p, sep="\t", index=False)
    monkeypatch.setattr("sys.argv", ["x", "--candidates", str(p), "--clindamycin-23s", "yes",
                                     "--out", str(tmp_path / "o"), "--n-boot", "50"])
    cb.main()
    v = json.loads((tmp_path / "o" / "verdict.json").read_text())
    assert v["isolates"] == 1500 and v["clindamycin_23s_in_rule"] == "yes"
    s = pd.read_csv(tmp_path / "o" / "scores.csv.gz", index_col=0)
    no_cluster = np.array([i % 13 == 0 for i in range(1500)])
    assert s.loc[no_cluster, "model|CIP"].isna().all()            # never scored without a cluster
    assert s.loc[~no_cluster, "model|CIP"].notna().all()
    d = cb.load(p, True)
    assert "tet(O)" in d["a_cols"] and "gyrA_T86I" in d["p_cols"]
    assert not any("PARTIAL" in c for c in d["X"].columns)        # PARTIAL calls are not counted
