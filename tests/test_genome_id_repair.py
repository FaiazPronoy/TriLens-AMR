from pathlib import Path

import pandas as pd
import pytest

from trilens.data.genome_id_repair import is_zero_padding, resolve

ROOT = Path(__file__).resolve().parents[1]


def table(rows):
    return pd.DataFrame(rows, columns=["strain_index", "genome_id", "sra", "size", "contigs"]).astype(str)


def records(rows):
    return pd.DataFrame(rows, columns=["genome_id", "sra_accession", "genome_length", "contigs"]).astype(str)


def test_trailing_zeros_restored_and_ambiguity_resolved_by_length_and_contigs():
    t = table([[0, "562.658", "ERR1", 4900000, 119], [1, "562.231", "ERR2", 5000000, 80], [2, "562.5", "ERR3", 1, 1]])
    r = records(
        [
            ["562.65800", "ERR1", 4900000, 119],
            ["562.2310", "ERR2", 5100000, 90],
            ["562.23100", "ERR2", 5000000, 80],
            ["562.5", "ERR3", 1, 1],
        ]
    )
    fixed, changed = resolve(t, r)
    assert fixed.genome_id.tolist() == ["562.65800", "562.23100", "562.5"]
    assert changed == 2


def test_multi_run_accession_field_is_split():
    t = table([[0, "562.7", "SRR9", 10, 2]])
    fixed, _ = resolve(t, records([["562.70", "SRR8,SRR9", 10, 2]]))
    assert fixed.genome_id.tolist() == ["562.70"]


def test_fails_on_no_match():
    t = table([[0, "562.658", "ERR1", 4900000, 119]])
    with pytest.raises(ValueError, match="1 strains with no match"):
        resolve(t, records([["562.65800", "ERR1", 4900001, 119]]))


def test_fails_on_two_matches():
    t = table([[0, "562.23", "ERR1", 10, 2]])
    with pytest.raises(ValueError, match="1 with several"):
        resolve(t, records([["562.230", "ERR1", 10, 2], ["562.2300", "ERR1", 10, 2]]))


def test_fails_when_id_changes_beyond_zeros():
    t = table([[0, "562.658", "ERR1", 10, 2]])
    with pytest.raises(ValueError, match="more than appended zeros"):
        resolve(t, records([["562.659", "ERR1", 10, 2]]))


def test_zero_padding_rule():
    assert is_zero_padding("562.658", "562.65800")
    assert is_zero_padding("562.5", "562.5")
    assert not is_zero_padding("562.658", "562.6581")
    assert not is_zero_padding("562", "5620")


def test_committed_ids_are_text_with_trailing_zeros():
    ids = pd.read_csv(ROOT / "results" / "data_inventory" / "genome_ids.csv", dtype=str)
    assert len(ids) == 6962
    assert ids.genome_id.is_unique
    assert ids.genome_id.str.fullmatch(r"562\.\d+").all()
    assert ids.genome_id.str.endswith("0").sum() == 706
