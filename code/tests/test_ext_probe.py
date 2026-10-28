"""ext_probe offline: our own strains and non-4.2.7 rows are excluded, AST is parsed per drug."""
import json
import sys

import pandas as pd

import src.mechanism.ext_probe as ep


def test_counts_external_candidates(tmp_path, monkeypatch):
    inv = tmp_path / "data_inventory"
    inv.mkdir()
    pd.DataFrame({"strain_index": [0, 1], "genome_id": ["562.1", "562.2"],
                  "sra": ["ERR_OURS", "ERR2"], "biosample": ["SAM_OURS", None]}).to_csv(
        inv / "genome_ids.csv", index=False)
    monkeypatch.setattr(ep, "inv", lambda: inv)
    table = pd.DataFrame({
        "Run": ["ERR_OURS", "SRR_A", "SRR_B", "SRR_C", "SRR_D"],
        "biosample_acc": ["SAM_OURS", "SAM_A", "SAM_B", "SAM_C", "SAM_D"],
        "geo_loc_name": ["UK:x", "USA:Boston", "USA:x", "Canada", None],
        "collection_date": ["2015", "2019", "2020", "2021", None],
        "epi_type": ["clinical", "clinical", None, "clinical", None],
        "isolation_source": [None, "blood", "chicken", "urine", None],
        "host": [None, "Homo sapiens", "Gallus gallus", None, None],
        "asm_stats_length_bp": ["5000000"] * 5, "asm_stats_n_contig": ["100"] * 5,
        "asm_stats_contig_n50": ["50000"] * 5, "number_drugs_tested": ["18"] * 5,
        "AST_phenotypes": ["ampicillin=R", "ampicillin=R,gentamicin=S,cefotaxime=I",
                           "ampicillin=S,gentamicin=R", None, "ampicillin=R"],
        "AMR_genotypes": ["blaTEM-1"] * 5,
        "amrfinder_version": ["4.2.7", "4.2.7", "4.2.7", "4.2.7", "3.11.26"],
        "refgene_db_version": ["2026-03-24.1"] * 5,
        "amrfinder_analysis_type": ["COMBINED"] * 5,
    })
    src = tmp_path / "PDG.1.metadata.tsv"
    table.to_csv(src, sep="\t", index=False)

    monkeypatch.setattr(ep, "newest_metadata_file", lambda: src.name)

    class FakeResp:
        def __init__(self, data):
            self._d = data

        def read(self):
            return self._d

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def urlopen(url, timeout=0):
        return FakeResp(src.read_bytes())

    monkeypatch.setattr(ep.urllib.request, "urlopen", urlopen)
    monkeypatch.setattr(ep.shutil, "copyfileobj", lambda r, f, length=0: f.write(r.read()))
    monkeypatch.setattr(sys, "argv", ["x", "--out", str(tmp_path / "out")])
    ep.main()

    s = json.loads((tmp_path / "out" / "ext_probe.json").read_text())
    assert s["rows_in_table"] == 5 and s["rows_with_AST"] == 4
    assert s["candidates"] == 2                      # ours out, the 3.11.26 row out, the no-AST row out
    per = {r["drug"]: r for r in s["per_drug"]}
    assert per["AMP"]["R"] == 1 and per["AMP"]["S"] == 1
    assert per["GEN"]["R"] == 1 and per["GEN"]["S"] == 1
    assert per["CTX"]["I"] == 1 and per["CTX"]["R"] == 0
    assert s["countries"] == {"USA": 2} and s["drugs_with_at_least_100_each"] == []


def test_parse_ast():
    assert ep.parse_ast("a=R,b=S, c=ND") == {"a": "R", "b": "S", "c": "ND"}
    assert ep.parse_ast(None) == {}
