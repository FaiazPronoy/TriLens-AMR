"""arg_fetch: shards, resume, failure limit, version check and the pre-registered matrix filters."""
import json
import os
import stat
import types

import pandas as pd
import pytest

import src.mechanism.arg_fetch as af

HEADER = ("Protein id\tContig id\tStart\tStop\tStrand\tElement symbol\tElement name\tScope\tType\tSubtype\t"
          "Class\tSubclass\tMethod\tTarget length\tReference sequence length\t% Coverage of reference\t"
          "% Identity to reference\tAlignment length\tClosest reference accession\tClosest reference name\t"
          "HMM accession\tHMM description\n")


def row(sym, typ="AMR", sub="AMR", ident=100, cov=100):
    return f"NA\tc1\t1\t9\t+\t{sym}\t{sym}\tcore\t{typ}\t{sub}\tX\tY\tEXACTX\t1\t1\t{cov}\t{ident}\t1\tA\tB\tNA\tNA\n"


@pytest.fixture
def env(tmp_path, monkeypatch):
    inv = tmp_path / "repo" / "data_inventory"
    inv.mkdir(parents=True)
    pd.DataFrame({"strain_index": range(8), "genome_id": [f"562.{i}" for i in range(8)]}).to_csv(
        inv / "genome_ids.csv", index=False)
    monkeypatch.setenv("TRILENS_REPO_RESULTS", str(tmp_path / "repo"))
    # fake amrfinder: version, database version, and a table chosen by the genome id
    tables = tmp_path / "tables"
    tables.mkdir()
    for i in range(8):
        rows = [row("blaCTX-M-15")] if i % 2 else [row("blaTEM-1")]
        rows += [row("gyrA_S83L", sub="POINT"), row("weak", ident=50)]
        (tables / f"562.{i}.tsv").write_text(HEADER + "".join(rows))
    fake = tmp_path / "bin" / "amrfinder"
    fake.parent.mkdir()
    fake.write_text(f"""#!/usr/bin/env bash
if [ "$1" = "--version" ]; then echo 4.2.7; exit 0; fi
if [ "$1" = "--database_version" ]; then echo "Database version: 2026-08-07.1"; exit 0; fi
fna="$2"; out="${{@: -1}}"; g=$(basename "$fna" .fna)
cp "{tables}/$g.tsv" "$out"
""")
    fake.chmod(fake.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("PATH", f"{fake.parent}:{os.environ['PATH']}")
    calls = []

    def fake_download(gid, dest, tries=3):
        calls.append(gid)
        if gid == "562.6":
            raise RuntimeError("HTTPError: 404")
        dest.write_bytes(b"A" * 2_000_000)
        return "https"

    monkeypatch.setattr(af, "download", fake_download)
    return tmp_path, calls


def args(tmp, **kw):
    d = dict(out=str(tmp / "calls"), shard=0, of=2, workers=2, threads=1, hours=1.0)
    d.update(kw)
    return types.SimpleNamespace(**d)


def test_fetch_resume_and_build(env):
    tmp, calls = env
    af.fetch(args(tmp, shard=1))            # strains 1, 3, 5, 7
    af.fetch(args(tmp, shard=0))            # 562.6 fails, is retried once in the same run, given up
    assert calls.count("562.6") == 2
    calls.clear()
    af.fetch(args(tmp, shard=0))            # a rerun touches nothing
    assert calls == []
    done = json.loads((tmp / "calls" / "shard_0" / "_DONE.json").read_text())
    assert done["ok"] == 3 and done["given_up"] == ["562.6"]

    out = tmp / "matrix"
    af.build(types.SimpleNamespace(calls=str(tmp / "calls"), out=str(out)))
    A = pd.read_csv(out / "acquired.csv", index_col=0)
    P = pd.read_csv(out / "point.csv", index_col=0)
    cov = json.loads((out / "coverage.json").read_text())
    assert cov["strains_with_table"] == 7 and cov["database"] == "2026-08-07.1"
    assert 6 not in A.index
    assert set(A.columns) == set()          # each symbol is in < 5 strains (4 and 3): dropped
    assert list(P.columns) == ["gyrA_S83L"] and "weak" not in P.columns


def test_build_refuses_mixed_versions(env):
    tmp, _ = env
    af.fetch(args(tmp, shard=1))
    (tmp / "calls" / "shard_0").mkdir()
    (tmp / "calls" / "shard_0" / "versions.json").write_text(json.dumps({"software": "4.0", "database": "x"}))
    (tmp / "calls" / "shard_0" / "tsv").mkdir()
    (tmp / "calls" / "shard_0" / "_DONE.json").write_text("{}")
    with pytest.raises(SystemExit, match="different"):
        af.build(types.SimpleNamespace(calls=str(tmp / "calls"), out=str(tmp / "m")))


def test_build_refuses_unfinished_shard(env):
    tmp, _ = env
    af.fetch(args(tmp, shard=1))
    (tmp / "calls" / "shard_0").mkdir()
    with pytest.raises(SystemExit, match="not finished"):
        af.build(types.SimpleNamespace(calls=str(tmp / "calls"), out=str(tmp / "m")))


def test_time_budget_stops_with_code_3(env):
    tmp, _ = env
    with pytest.raises(SystemExit) as e:
        af.fetch(args(tmp, shard=1, hours=0.0))
    assert e.value.code == 3
