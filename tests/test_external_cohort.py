import numpy as np
import pandas as pd
import pytest

pytest.importorskip("xgboost")
import trilens.experiments.external_cohort as rx


@pytest.mark.parametrize(
    "drug,genes,points,expected",
    [
        ("AMP", {"blaTEM-1"}, set(), 1.0),
        ("AMP", {"blaEC-5"}, set(), 0.0),
        ("CTX", {"blaTEM-1"}, set(), 0.0),
        ("CTX", {"blaTEM-52"}, set(), 1.0),
        ("CTX", {"blaSHV-1"}, set(), 0.0),
        ("CTX", {"blaSHV-12"}, set(), 1.0),
        ("CAZ", {"blaCTX-M-15"}, set(), 1.0),
        ("CAZ", {"blaCMY-2"}, set(), 1.0),
        ("GEN", {"aac(3)-IId"}, set(), 1.0),
        ("GEN", {"aac(6')-Ib-cr5"}, set(), 0.0),
        ("SXT", {"sul1"}, set(), 0.0),
        ("SXT", {"dfrA17"}, set(), 1.0),
        ("CIP", set(), {"gyrA_S83L"}, 1.0),
        ("CIP", {"gyrA_S83L"}, set(), 0.0),
        ("CIP", {"qnrS1"}, set(), 1.0),
        ("AMC", {"blaOXA-1"}, set(), 1.0),
        ("AMC", {"blaOXA-10"}, set(), 0.0),
        ("AMC", {"blaTEM-1"}, set(), 0.0),
    ],
)
def test_rules(drug, genes, points, expected):
    assert rx.rule_score(genes, points, drug) == expected


def test_calls_and_ast_parsing():
    a, p = rx.calls("blaTEM-1,gyrA_S83L=POINT,sul1=PARTIAL,tet(A)")
    assert a == {"blaTEM-1", "tet(A)"} and p == {"gyrA_S83L"}
    with pytest.raises(SystemExit):
        rx.calls("blaX=NEWTAG")
    s = "ampicillin=R,gentamicin=I,ceftazidime=ND"
    assert rx.ast(s, "AMP", "ri") == 1.0
    assert rx.ast(s, "GEN", "ri") == 1.0 and np.isnan(rx.ast(s, "GEN", "no_intermediate"))
    assert np.isnan(rx.ast(s, "CAZ", "ri")) and np.isnan(rx.ast(s, "CIP", "ri"))


def _world(n, rng, signal=True):
    g = {k: rng.integers(0, 2, n) for k in ["blaCTX-M-15", "aac(3)-IId", "blaTEM-1", "dfrA17", "blaOXA-1"]}
    gy = rng.integers(0, 2, n)

    def flip(p):
        return (rng.random(n) < p).astype(int)

    lab = {
        "AMP": g["blaTEM-1"],
        "CTX": g["blaCTX-M-15"],
        "CAZ": g["blaCTX-M-15"],
        "GEN": g["aac(3)-IId"],
        "SXT": g["dfrA17"],
        "AMC": g["blaOXA-1"],
        "CIP": gy,
    }
    if not signal:
        lab = {k: rng.integers(0, 2, n) for k in lab}
    lab = {k: np.abs(v - flip(0.05)) for k, v in lab.items()}
    return g, gy, lab


def test_transfer_passes_when_genes_carry_the_signal(tmp_path, monkeypatch):
    monkeypatch.setitem(rx.XGB, "n_estimators", 30)
    rng = np.random.default_rng(0)
    g, gy, lab = _world(800, rng)
    train = {
        "A": pd.DataFrame(g, dtype="uint8"),
        "P": pd.DataFrame({"gyrA_S83L": gy}, dtype="uint8"),
        "M": pd.DataFrame({c: rng.random(800) for c in rx.ASM_OURS}),
        "Y": pd.DataFrame({d: lab[d].astype(float) for d in rx.PD_NAME}),
    }
    g2, gy2, lab2 = _world(600, rng)
    rows = []
    for i in range(600):
        genes = [k for k in g2 if g2[k][i]] + (["gyrA_S83L=POINT"] if gy2[i] else []) + ["blaNEW-1"]
        ast = ",".join(f"{rx.PD_NAME[d]}={'R' if lab2[d][i] else 'S'}" for d in rx.PD_NAME)
        rows.append(
            dict(
                AMR_genotypes=",".join(genes),
                AST_phenotypes=ast,
                amrfinder_version="4.2.7",
                refgene_db_version=rx.PRIMARY_DB,
                bioproject_acc=f"PRJ{i % 40}",
                asm_stats_length_bp=str(5_000_000 + i),
                asm_stats_n_contig="100",
                asm_stats_contig_n50="50000",
            )
        )
    p = tmp_path / "cand.tsv.gz"
    pd.DataFrame(rows).to_csv(p, sep="\t", index=False)
    ext = rx.load_external(p, "primary", "ri", list(train["A"].columns), list(train["P"].columns))
    assert ext["unseen_genes"] == ["blaNEW-1"] and ext["n"] == 600
    m, dl, v = rx.run(train, ext, tmp_path / "out", n_boot=100)
    assert v["primary_pass"] and v["bioprojects"] == 40
    assert (tmp_path / "out" / "verdict.json").exists()


def test_transfer_fails_without_signal(tmp_path, monkeypatch):
    monkeypatch.setitem(rx.XGB, "n_estimators", 30)
    rng = np.random.default_rng(1)
    g, gy, lab = _world(500, rng, signal=False)
    train = {
        "A": pd.DataFrame(g, dtype="uint8"),
        "P": pd.DataFrame({"gyrA_S83L": gy}, dtype="uint8"),
        "M": pd.DataFrame({c: rng.random(500) for c in rx.ASM_OURS}),
        "Y": pd.DataFrame({d: lab[d].astype(float) for d in rx.PD_NAME}),
    }
    g2, gy2, lab2 = _world(400, rng, signal=False)
    rows = [
        dict(
            AMR_genotypes=",".join(k for k in g2 if g2[k][i]),
            AST_phenotypes=",".join(f"{rx.PD_NAME[d]}={'R' if lab2[d][i] else 'S'}" for d in rx.PD_NAME),
            amrfinder_version="4.2.7",
            refgene_db_version=rx.PRIMARY_DB,
            bioproject_acc=f"P{i % 30}",
            asm_stats_length_bp="5000000",
            asm_stats_n_contig="100",
            asm_stats_contig_n50="50000",
        )
        for i in range(400)
    ]
    p = tmp_path / "c.tsv.gz"
    pd.DataFrame(rows).to_csv(p, sep="\t", index=False)
    ext = rx.load_external(p, "primary", "ri", list(train["A"].columns), list(train["P"].columns))
    *_, v = rx.run(train, ext, tmp_path / "o", n_boot=50)
    assert not v["primary_pass"]
