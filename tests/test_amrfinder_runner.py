from pathlib import Path

import pandas as pd
import pytest

from trilens.determinants import amrfinder_runner as am

FIXTURE = Path(__file__).parent / "data" / "amrfinder_test_dna.expected"


def test_parse_ncbi_test_output():
    hits = am.parse_hits(FIXTURE)
    assert len(hits) == 24 and hits.locus.notna().all()
    assert {"blaTEM-156", "sul2", "blaEC"} <= set(hits.symbol)
    assert hits.loc[hits.symbol == "blaTEMp_G162T", "is_point"].all()


def test_parse_accepts_version_3_names(tmp_path):
    p = tmp_path / "v3.tsv"
    pd.DataFrame(
        {
            "Protein identifier": ["NA"],
            "Contig id": ["LMHECDEF_00001"],
            "Gene symbol": ["dfrA1"],
            "Sequence name": ["trimethoprim-resistant dihydrofolate reductase DfrA1"],
            "Element type": ["AMR"],
            "Element subtype": ["AMR"],
            "Class": ["TRIMETHOPRIM"],
            "Subclass": ["TRIMETHOPRIM"],
            "% Coverage of reference sequence": ["100.00"],
            "% Identity to reference sequence": ["100.00"],
        }
    ).to_csv(p, sep="\t", index=False)
    d = am.determinants(am.parse_hits(p))
    assert list(d.drug) == ["SXT"] and d.counted.all() and d.locus.iloc[0] == "LMHECDEF_00001"


@pytest.mark.parametrize(
    "cls, sub, expected",
    [
        ("BETA-LACTAM", "BETA-LACTAM", {"AMP", "AMC"}),
        ("BETA-LACTAM", "CEPHALOSPORIN", {"AMP", "AMC", "CTX", "CAZ"}),
        ("QUINOLONE", "QUINOLONE", {"CIP"}),
        ("AMINOGLYCOSIDE", "GENTAMICIN", {"GEN"}),
        ("AMINOGLYCOSIDE", "STREPTOMYCIN", set()),
        ("SULFONAMIDE", "SULFONAMIDE", {"SXT"}),
        ("COLISTIN", "COLISTIN", set()),
    ],
)
def test_drug_mapping(cls, sub, expected):
    assert set(am.drugs_for(cls, sub)) == expected


def test_intrinsic_and_cutoff_rules():
    hits = am.parse_hits(FIXTURE)
    loci = dict(zip(hits.symbol, hits.locus))
    prev = {loci["blaEC"]: 1.0, loci["blaTEM-156"]: 0.4}
    d = am.determinants(hits, prev)
    ec = d[d.symbol == "blaEC"]
    assert ec.intrinsic.all() and not ec.counted.any()
    tem = d[d.symbol == "blaTEM-156"]
    assert tem.counted.all() and set(tem.drug) == {"AMP", "AMC"}
    assert not d[d.symbol == "blaTEMp_G162T"].intrinsic.any()
    assert set(d[d.symbol == "aph(3'')-Ib"].drug) == set()
