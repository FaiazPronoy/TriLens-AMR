"""The journal-extension Kaggle stages (kaggle/make_nbs.py) and the pieces they run."""
import json
import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from src.baselines.snp_distance import snp_hamming
from src.data.metadata import PROCESSED, ROOT
from src.splits import from_json

REPO = ROOT.parent
NB_DIR = REPO / "kaggle" / "notebooks"
NEW = ("w1_gene_sets", "w1_classical", "w2_snp_knn", "w4_amrfinder")
HAVE_SPLITS = all((PROCESSED / f"{n}_split.pkl").exists()
                  for n in ("country_uk_to_norway", "country_norway_to_uk", "time_norway_2013"))


def _cells(name):
    nb = json.loads((NB_DIR / f"{name}.ipynb").read_text(encoding="utf-8"))
    return ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]


def test_committed_notebooks_are_the_generator_output(tmp_path):
    subprocess.run([sys.executable, str(REPO / "kaggle" / "make_nbs.py"), str(tmp_path)], check=True,
                   capture_output=True)
    for f in tmp_path.glob("*.ipynb"):
        assert f.read_text(encoding="utf-8") == (NB_DIR / f.name).read_text(encoding="utf-8"), \
            f"{f.name} is stale: run python kaggle/make_nbs.py kaggle/notebooks"


@pytest.mark.parametrize("name", NEW)
def test_new_stage_is_wired(name):
    setup, *rest = _cells(name)
    assert f'NAME   = "{name}"' in setup
    writes = json.loads(setup.split("WRITES = ")[1].split("\n")[0])
    steps = [c for c in rest if c.startswith("run(")]
    assert steps, "no steps"
    for c in steps:                                     # every marker lies inside what the stage writes
        if "\n" in c:
            marker = c.split('"')[-2]
            assert marker.split("/")[0] in writes, (name, marker)
    cpu = name != "w1_classical"
    assert ("NEEDS_GPU = False" in setup) == cpu
    if name.startswith(("w1", "w2")):                   # splits come from the committed JSON, not uploads
        assert "src.splits.from_json" in steps[0]
    assert "TRILENS_REPO_RESULTS" in setup
    assert 'CODE_REF    = "extension"' in setup         # the extension code is not on main


def test_w1_classical_needs_the_gene_sets():
    setup = _cells("w1_classical")[0]
    assert 'NEEDS  = ["w1_gene_sets"]' in setup


def test_split_from_json_rejects_overlap():
    d = {"name": "x", "rules": {}, "antibiotics": {}, "info": {},
         "train_idx": [0, 1], "val_idx": [2], "test_idx": [2, 3], "unused_idx": []}
    with pytest.raises(ValueError):
        from_json.split_from_json(d, 4, [None] * 4)


@pytest.mark.skipif(not HAVE_SPLITS, reason="split files not built (python -m src.splits.country_time)")
def test_split_from_json_equals_the_builder():
    with open(PROCESSED / "mlst_lineage_holdout_split.pkl", "rb") as f:
        st = pickle.load(f)["st_per_genome"]
    for p in sorted((ROOT / "results" / "splits").glob("*.json")):
        d = json.loads(p.read_text())
        rebuilt = from_json.split_from_json(d, len(st), st)
        with open(PROCESSED / f"{d['name']}_split.pkl", "rb") as f:
            built = pickle.load(f)
        for k in ("train_idx", "val_idx", "test_idx", "unused_idx"):
            assert np.array_equal(rebuilt[k], built[k]), (p.name, k)


def test_snp_hamming_matches_brute_force():
    rng = np.random.default_rng(0)
    X = rng.choice(np.array([0, 0, 0, 1, 2, 3], dtype=np.int8), size=(25, 103))
    D = snp_hamming(X, chunk=17)
    nz = X != 0
    assert np.array_equal(D, (nz[:, None, :] != nz[None, :, :]).sum(-1))


@pytest.mark.skipif(not HAVE_SPLITS, reason="split files not built")
def test_split_pipeline_points_the_scripts_at_the_split():
    from src.experiments import run_split_pipeline as rsp
    m = rsp.configure("classical", "time_norway_2013", "cpu")
    assert m.SPLIT_KEY == "time_norway_2013" and m.REQUIRE_ST_DISJOINT is False
    assert m.ANTIBIOTICS == ["AMP", "CIP", "SXT", "GEN", "CTX", "CAZ"]
    assert m.XGB_PARAMS["device"] == "cpu" and m.OUTPUT_DIR.name == "time_norway_2013"
    r = rsp.configure("rank", "country_norway_to_uk", "cpu")
    assert r.SPLIT_PKL.name == "country_norway_to_uk_split.pkl" and "AMC" not in r.ANTIBIOTICS
    b = rsp.configure("build", "country_uk_to_norway", "cpu")
    assert b.OUTPUT_PKL.name == "arcadia_v2_gene_filtered__country_uk_to_norway.pkl"
    assert b.SPLIT_KEY == "country_uk_to_norway"
