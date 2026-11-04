import functools
import http.server
import json
import threading

import numpy as np
import pandas as pd
import pytest

import trilens.determinants.species_screen as sp


def _clusters(n, clusters=60):
    return pd.DataFrame(
        {"PDS_acc": [f"PDS{i % clusters}" for i in range(n)], "target_acc": [f"PDT{i}" for i in range(n)]}
    )


def _meta(n, rng, clusters=60, drugs=("ampicillin", "gentamicin", "ciprofloxacin", "cefotaxime")):
    rows = []
    for i in range(n):
        ast = ",".join(f"{d}={'R' if rng.random() < 0.3 else 'S'}" for d in drugs)
        rows.append(
            dict(
                target_acc=f"PDT{i}",
                biosample_acc=f"SAMN{i}",
                AST_phenotypes=ast if i % 10 else None,
                AMR_genotypes="blaTEM-1",
                amrfinder_version="4.2.7",
                refgene_db_version="2026-03-24.1",
                asm_stats_length_bp="5000000",
                asm_stats_n_contig="80",
                asm_stats_contig_n50="90000",
                geo_loc_name="USA: X",
                epi_type="clinical",
            )
        )
    return pd.DataFrame(rows)


@pytest.fixture
def ftp(tmp_path):
    rng = np.random.default_rng(0)
    root = tmp_path / "Results"
    for org, n in [("Klebsiella", 3000), ("Salmonella", 4000), ("Tiny", 300)]:
        d = root / org / "latest_snps" / "Metadata"
        d.mkdir(parents=True)
        _meta(n, rng).to_csv(d / f"PDG000.{n}.metadata.tsv", sep="\t", index=False)
        if org != "Salmonella":
            c = root / org / "latest_snps" / "Clusters"
            c.mkdir()
            _clusters(n).to_csv(c / f"PDG000.{n}.reference_target.cluster_list.tsv", sep="\t", index=False)
    (root / "Escherichia_coli_Shigella" / "latest_snps" / "Metadata").mkdir(parents=True)
    (root / "NoSnps").mkdir()
    h = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(tmp_path))
    h.log_message = lambda *a, **k: None
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), h)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}/Results/", tmp_path
    srv.shutdown()


def test_probe_counts_gates_and_chooses_by_count(ftp, monkeypatch, tmp_path):
    base, _ = ftp
    out = tmp_path / "out"
    monkeypatch.setattr("sys.argv", ["x", "--out", str(out), "--base", base, "--tmp", str(tmp_path)])
    sp.main()
    recs = {r["organism"]: r for r in json.loads((out / "per_species.json").read_text())}
    assert "Escherichia_coli_Shigella" not in recs
    assert "error" in recs["NoSnps"] and not recs["NoSnps"]["passes"]
    k = recs["Klebsiella"]
    assert k["rows_in_table"] == 3000 and k["ast_with_amrfinder_4_2_7"] == 2700
    assert k["cluster_share"] == 1.0 and k["clusters"] == 54 and k["passes"]
    assert not recs["Tiny"]["passes"] and not recs["Tiny"]["gate"]["isolates"]
    s = recs["Salmonella"]
    assert s["clusters"] == 0 and not s["passes"] and "HTTP 404" in s["cluster_source"]["error"]
    assert k["cluster_source"]["cluster_column"] == "PDS_acc"
    gate = pd.read_csv(out / "gate.csv").set_index("organism")
    assert gate.loc["Klebsiella", "chosen"] and not gate.loc["Salmonella", "chosen"]
    assert (out / "candidates_Klebsiella.tsv.gz").exists() and (out / "cluster_source.json").exists()


def test_gate_fails_on_too_few_clusters(ftp, monkeypatch, tmp_path):
    base, root = ftp
    d = root / "Results" / "FewClusters" / "latest_snps" / "Metadata"
    d.mkdir(parents=True)
    _meta(3000, np.random.default_rng(1)).to_csv(d / "PDG1.metadata.tsv", sep="\t", index=False)
    c = d.parent / "Clusters"
    c.mkdir()
    _clusters(3000, clusters=10).to_csv(c / "PDG1.reference_target.cluster_list.tsv", sep="\t", index=False)
    out = tmp_path / "o2"
    monkeypatch.setattr(
        "sys.argv", ["x", "--out", str(out), "--base", base, "--only", "FewClusters", "--tmp", str(tmp_path)]
    )
    sp.main()
    r = json.loads((out / "per_species.json").read_text())[0]
    assert r["clusters"] == 9 and not r["gate"]["clusters"] and not r["passes"]
