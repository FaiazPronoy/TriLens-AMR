import types

import pandas as pd

from trilens.determinants import call_concordance as cc


def hits(rows):
    return pd.DataFrame(rows, columns=["strain_index", "symbol", "type", "method", "identity", "coverage", "is_point"])


def test_concordance_counts_and_drop(tmp_path):
    h = hits(
        [
            [0, "blaTEM-1", "AMR", "EXACTX", 100, 100, False],
            [0, "emrD", "AMR", "EXACTX", 100, 100, False],
            [0, "gyrA_S83L", "AMR", "POINTX", 100, 100, True],
            [1, "sul2", "AMR", "PARTIAL_CONTIG_ENDX", 100, 60, False],
        ]
    )
    hp = tmp_path / "h.csv"
    h.to_csv(hp, index=False)
    j = pd.DataFrame(
        {
            "strain_index": ["0", "1"],
            "AMR_genotypes": ["blaTEM-1,gyrA_S83L=POINT", "sul2"],
            "amrfinder_version": ["4.2.7", "4.2.7"],
            "amrfinder_analysis_type": ["COMBINED", "NUCLEOTIDE"],
        }
    )
    jp = tmp_path / "j.tsv"
    j.to_csv(jp, sep="\t", index=False)
    out = tmp_path / "c.json"
    cc.run(types.SimpleNamespace(hits=str(hp), joined=str(jp), software="4.2.7", out=str(out)))
    import json

    r = json.loads(out.read_text())
    assert r["overlap"] == 2
    assert r["acquired"]["calls_only_ours"] == 1 and r["acquired"]["calls_only_ncbi"] == 1
    assert r["acquired_without_emrD"]["calls_only_ours"] == 0
    assert r["point"]["pair_agreement"] == 1.0
