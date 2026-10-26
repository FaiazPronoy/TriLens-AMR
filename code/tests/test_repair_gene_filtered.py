import json
import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "repair_gene_filtered.py"


def _run(tmp_path, X_file, pos_file, X_snp, pos_snp):
    src, snp = tmp_path / "gf.pkl", tmp_path / "snp.pkl"
    with open(src, "wb") as f:
        pickle.dump({"X_snp_filtered": X_file, "snp_positions_filtered": pos_file, "gene_to_range": {}}, f)
    with open(snp, "wb") as f:
        pickle.dump({"X_snp": X_snp, "snp_positions": pos_snp}, f)
    r = subprocess.run([sys.executable, str(SCRIPT), "--src", str(src), "--snp", str(snp),
                        "--out", str(tmp_path / "out.pkl"), "--report", str(tmp_path / "rep.json")],
                       capture_output=True, text=True)
    return r.returncode, json.loads((tmp_path / "rep.json").read_text())


def _matrix(n=6, m=10, seed=0):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 4, size=(n, m)).astype(np.int8), np.array([f"p{i}" for i in range(m)])


def test_clean_file_identical_to_matrix(tmp_path):
    X, pos = _matrix()
    code, rep = _run(tmp_path, X[:, 2:8].copy(), pos[2:8], X, pos)
    assert code == 2


def test_in_range_damage_is_refused(tmp_path):
    X, pos = _matrix()
    bad = X[:, 2:8].copy()
    bad[:, 1] = (bad[:, 1] + 1) % 4
    code, rep = _run(tmp_path, bad, pos[2:8], X, pos)
    assert code == 1 and "different from the SNP matrix" in rep["status"]


def test_out_of_range_damage_is_repaired(tmp_path):
    X, pos = _matrix()
    bad = X[:, 2:8].copy()
    bad[:, 3] = -100
    code, rep = _run(tmp_path, bad, pos[2:8], X, pos)
    assert code == 0
    with open(tmp_path / "out.pkl", "rb") as f:
        assert np.array_equal(pickle.load(f)["X_snp_filtered"], X[:, 2:8])


def test_ambiguous_position_names_are_refused(tmp_path):
    X, pos = _matrix()
    pos = pos.copy()
    pos[9] = pos[4]
    bad = X[:, 2:8].copy()
    bad[:, 2] = 120
    code, rep = _run(tmp_path, bad, pos[2:8], X, pos)
    assert code == 1 and rep["ambiguous_position_names"] == ["p4"]
