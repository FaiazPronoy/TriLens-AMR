"""arg_probe2 offline: fake Pathogen Detection listing and tables, fake BV-BRC API."""
import json
import sys

import pandas as pd

import src.mechanism.arg_probe2 as p2


def test_probe2_joins_by_run_and_biosample(tmp_path, monkeypatch):
    inv = tmp_path / "repo" / "data_inventory"
    inv.mkdir(parents=True)
    pd.DataFrame({"strain_index": [0, 1, 2], "genome_id": ["562.1", "562.2", "562.3"],
                  "sra": ["ERR1", "ERR2", "ERR3"], "biosample": ["SAMEA1", None, "SAMEA3"]}).to_csv(
        inv / "genome_ids.csv", index=False)
    monkeypatch.setattr(p2, "_INV", inv)
    meta = pd.DataFrame({"Run": ["ERR1", "ERRx", "ERR2"], "biosample_acc": ["SAMEA1", "SAMx", "SAMEA2"],
                         "AMR_genotypes": ["blaCTX-M-15=COMPLETE,sul1=COMPLETE", "", "blaTEM-1=COMPLETE"],
                         "amrfinder_version": ["3.11.2", "3.12.8", "3.11.2"]})
    amr = pd.DataFrame({"biosample_acc": ["SAMEA3", "SAMEA3"], "element_symbol": ["aac(3)-IId", "dfrA17"],
                        "amrfinderplus_db_version": ["2023-04-17.1", "2023-04-17.1"]})
    src = tmp_path / "src"
    src.mkdir()
    meta.to_csv(src / "PDG4.1.metadata.tsv", sep="\t", index=False)
    amr.to_csv(src / "PDG4.1.amr.metadata.tsv", sep="\t", index=False)

    def get(url, timeout=120):
        if url.endswith("Metadata/"):
            return 200, b'<a href="PDG4.1.metadata.tsv">x</a><a href="PDG4.1.metadata.xml">y</a>', ""
        if url.endswith("AMR/"):
            return 200, b'<a href="PDG4.1.amr.metadata.tsv">x</a>', ""
        if "bv-brc" in url:
            return "URLError", b"", "URLError: <urlopen error timed out>"
        raise AssertionError(url)

    def download(url, dest):
        dest.write_bytes((src / url.rsplit("/", 1)[1]).read_bytes())
        return 0.1, 1

    monkeypatch.setattr(p2, "get", get)
    monkeypatch.setattr(p2, "download", download)
    monkeypatch.setattr(sys, "argv", ["x", "--out", str(tmp_path / "out")])
    p2.main()
    s = json.loads((tmp_path / "out" / "probe2.json").read_text())
    m, a = s["pathogen_detection_metadata"], s["pathogen_detection_amr"]
    assert m["strains_covered"] == 2 and m["strains_covered_by_sra"] == 2 and m["rows_in_file"] == 3
    assert m["version_columns"]["amrfinder_version"] == {"3.11.2": 2}
    assert m["family_prevalence_in_joined_rows"]["CTX-M"] == 0.5
    assert a["strains_covered"] == 1 and a["joined_rows"] == 2
    assert s["bvbrc_api"][0]["error"].startswith("URLError")
    assert (tmp_path / "out" / "metadata_joined.tsv.gz").exists()
