import json

import pandas as pd
import pytest

import trilens.determinants.determinant_matrix as ap


def test_committed_table_matches_its_hash_and_counts(tmp_path):
    ap.build(ap.PRIMARY_DB, tmp_path / "m")
    cov = json.loads((tmp_path / "m" / "coverage.json").read_text())
    assert cov["strains_with_table"] == 5072 and cov["software"] == "4.2.7"
    A = pd.read_csv(tmp_path / "m" / "acquired.csv", index_col=0)
    assert A.index.name == "strain_index" and A.shape[0] == 5072 and (A.sum() >= ap.MIN_STRAINS).all()
    assert "blaCTX-M-15" in A.columns and not any("=" in c for c in A.columns)
    P = pd.read_csv(tmp_path / "m" / "point.csv", index_col=0)
    assert any(c.startswith("gyrA_") for c in P.columns)
    ap.build("all", tmp_path / "all")
    assert json.loads((tmp_path / "all" / "coverage.json").read_text())["strains_with_table"] == 5223


def test_rules_on_a_small_table(tmp_path, monkeypatch):
    inv = tmp_path / "data_inventory"
    inv.mkdir()
    monkeypatch.setattr(ap, "inv", lambda: inv)
    pd.DataFrame(
        {
            "strain_index": range(4),
            "country": ["UK"] * 4,
            "year": [2010] * 4,
            "study": ["s"] * 4,
            "st": ["1"] * 4,
            "phylogroup": ["B2"] * 4,
        }
    ).to_csv(inv / "strain_table.csv", index=False)
    rows = pd.DataFrame(
        {
            "strain_index": range(4),
            "amrfinder_version": ["4.2.7"] * 3 + ["3.11.26"],
            "refgene_db_version": [ap.PRIMARY_DB] * 4,
            "amrfinder_analysis_type": ["NUCLEOTIDE"] * 4,
            "AMR_genotypes": ["blaCTX-M-15,gyrA_S83L=POINT,sul1=PARTIAL"] * 4,
        }
    )
    rows.to_csv(inv / "pd_amr_joined.tsv.gz", sep="\t", index=False, compression={"method": "gzip", "mtime": 0})
    import hashlib

    (inv / "pd_amr_joined.json").write_text(
        json.dumps(
            {"source_file": "x", "sha256": hashlib.sha256((inv / "pd_amr_joined.tsv.gz").read_bytes()).hexdigest()}
        )
    )
    monkeypatch.setattr(ap, "MIN_STRAINS", 1)
    ap.build(ap.PRIMARY_DB, tmp_path / "m")
    A = pd.read_csv(tmp_path / "m" / "acquired.csv", index_col=0)
    P = pd.read_csv(tmp_path / "m" / "point.csv", index_col=0)
    assert list(A.columns) == ["blaCTX-M-15"] and list(P.columns) == ["gyrA_S83L"] and len(A) == 3
    rows.loc[0, "AMR_genotypes"] = "blaX=NEWTAG"
    rows.to_csv(inv / "pd_amr_joined.tsv.gz", sep="\t", index=False, compression={"method": "gzip", "mtime": 0})
    with pytest.raises(SystemExit, match="changed since it was frozen"):
        ap.build(ap.PRIMARY_DB, tmp_path / "m2")
    (inv / "pd_amr_joined.json").write_text(
        json.dumps(
            {"source_file": "x", "sha256": hashlib.sha256((inv / "pd_amr_joined.tsv.gz").read_bytes()).hexdigest()}
        )
    )
    with pytest.raises(SystemExit, match="unknown AMR_genotypes suffix"):
        ap.build(ap.PRIMARY_DB, tmp_path / "m3")
