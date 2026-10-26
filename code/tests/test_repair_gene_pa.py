"""repair_gene_pa.py: rebuilds presence from the .bim allele letters, nothing else."""
import json
import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "repair_gene_pa.py"


def write_plink(tmp, truth, present_letter="2", missing=()):
    """truth: (n_samples, n_loci) 0/1. A1 = the minority state, as PLINK does by default."""
    n, m = truth.shape
    absent_letter = "1" if present_letter == "2" else "2"
    a1, a2, codes = [], [], np.zeros((m, n), dtype=np.uint8)
    for j in range(m):
        minor_is_present = truth[:, j].mean() <= 0.5
        l1, l2 = (present_letter, absent_letter) if minor_is_present else (absent_letter, present_letter)
        a1.append(l1); a2.append(l2)
        carries_a1 = truth[:, j] == (1 if minor_is_present else 0)
        codes[j] = np.where(carries_a1, 0b00, 0b11)
    for (i, j) in missing:
        codes[j, i] = 0b01
    per = (n + 3) // 4
    body = np.zeros((m, per), dtype=np.uint8)
    padded = np.zeros((m, per * 4), dtype=np.uint8); padded[:, :n] = codes
    for k in range(4):
        body |= (padded[:, k::4] & 0b11) << (2 * k)
    bed = tmp / "pa.bed"
    with open(bed, "wb") as f:
        f.write(bytes([0x6C, 0x1B, 0x01])); f.write(body.tobytes())
    ids = [f"G{j:03d}" for j in range(m)]
    pd.DataFrame({"c": ids, "i": ".", "cm": 0, "p": 1, "a1": a1, "a2": a2}).to_csv(
        tmp / "pa.bim", sep="\t", header=False, index=False)
    sras = [f"SRR{i}" for i in range(n)]
    pd.DataFrame({"f": sras, "i": sras, "a": 0, "b": 0, "c": 0, "d": -9}).to_csv(
        tmp / "pa.fam", sep=" ", header=False, index=False)
    pd.DataFrame({"SRA.Accession": sras, "Genome.Name": [f"strain {i}" for i in range(n)]}).to_csv(
        tmp / "sra.csv", index=False)
    old = (codes.T == 0b00).astype(np.uint8)       # the old rule: present = (PLINK value == 0)
    old[codes.T == 0b01] = 1                       # NaN was filled with 0, i.e. "present"
    return ids, old


def setup(tmp, truth, **kw):
    ids, old = write_plink(tmp, truth, **kw)
    n = truth.shape[0]
    pkl = {"X_pa": old, "gene_ids": ids, "sample_genome_names": [f"strain {i}" for i in range(n)],
           "matched_mask": np.ones(n, dtype=bool), "encoding_rule": "present = (PLINK_value == 0)"}
    with open(tmp / "old.pkl", "wb") as f:
        pickle.dump(pkl, f)
    ecor = (truth.mean(0) * 72).round().astype(int)
    ecor[: truth.shape[1] // 3] = 72                                    # first third are core loci
    pd.DataFrame({"gene_id": ids, "kind": "cds", "ecor_cluster": ids, "ecor_genomes": ecor}).to_csv(
        tmp / "ecor.csv", index=False)
    return ids


def run(tmp):
    r = subprocess.run([sys.executable, str(SCRIPT), "--bed", str(tmp / "pa.bed"), "--old", str(tmp / "old.pkl"),
                        "--sra", str(tmp / "sra.csv"), "--out", str(tmp / "new.pkl"),
                        "--report", str(tmp / "rep.json"), "--ecor", str(tmp / "ecor.csv")],
                       capture_output=True, text=True)
    return r.returncode, json.loads((tmp / "rep.json").read_text())


def truth_matrix(seed=0, n=41, m=30):
    rng = np.random.default_rng(seed)
    t = np.zeros((n, m), dtype=np.uint8)
    t[:, : m // 3] = rng.random((n, m // 3)) < 0.97                    # core: present almost everywhere
    p = rng.uniform(0.02, 0.8, m - m // 3)
    t[:, m // 3:] = rng.random((n, m - m // 3)) < p
    return t


def test_recovers_presence(tmp_path):
    t = truth_matrix()
    setup(tmp_path, t)
    code, rep = run(tmp_path)
    assert code == 0, rep["status"]
    new = pickle.load(open(tmp_path / "new.pkl", "rb"))["X_pa"]
    assert np.array_equal(new, t)
    assert rep["present_letter"] == "2" and rep["loci_flipped"] > 0 and rep["loci_inconsistent"] == 0


def test_other_present_letter(tmp_path):
    t = truth_matrix(1)
    setup(tmp_path, t, present_letter="1")
    code, rep = run(tmp_path)
    assert code == 0 and rep["present_letter"] == "1"
    assert np.array_equal(pickle.load(open(tmp_path / "new.pkl", "rb"))["X_pa"], t)


def test_missing_calls_become_absent(tmp_path):
    t = truth_matrix(2)
    setup(tmp_path, t, missing=[(0, 0), (3, 12)])
    code, rep = run(tmp_path)
    assert code == 0 and rep["missing_cells"] == 2
    new = pickle.load(open(tmp_path / "new.pkl", "rb"))["X_pa"]
    assert new[0, 0] == 0 and new[3, 12] == 0


def test_refuses_more_than_a_flip(tmp_path):
    t = truth_matrix(3)
    setup(tmp_path, t)
    old = pickle.load(open(tmp_path / "old.pkl", "rb"))
    old["X_pa"][5, 20] ^= 1                                              # one cell that is not a flip
    pickle.dump(old, open(tmp_path / "old.pkl", "wb"))
    code, rep = run(tmp_path)
    assert code == 1 and "more than a flip" in rep["status"] and not (tmp_path / "new.pkl").exists()
