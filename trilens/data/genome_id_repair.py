from __future__ import annotations

import argparse
import json
import time
import urllib.request
from pathlib import Path

import pandas as pd

API = "https://www.bv-brc.org/api/genome/"
FIELDS = "genome_id,genome_name,sra_accession,biosample_accession,genome_length,contigs"
BATCH = 200
ROOT = Path(__file__).resolve().parents[2]
TABLE = ROOT / "results" / "data_inventory" / "genome_ids.csv"


def query(rql, tries=4):
    body = f"{rql}&select({FIELDS})&limit(25000)".encode()
    req = urllib.request.Request(
        API,
        data=body,
        headers={
            "Accept": "application/json",
            "Content-Type": "application/rqlquery+x-www-form-urlencoded",
            "User-Agent": "trilens-amr",
        },
    )
    for attempt in range(tries):
        try:
            with urllib.request.urlopen(req, timeout=180) as r:
                return json.loads(r.read())
        except Exception:
            if attempt == tries - 1:
                raise
            time.sleep(10 * (attempt + 1))


def fetch_records(runs, batch=BATCH):
    runs = list(dict.fromkeys(runs))
    out = []
    for i in range(0, len(runs), batch):
        chunk = runs[i : i + batch]
        out += query(f"in(sra_accession,({','.join(chunk)}))")
        print(f"  queried {min(i + batch, len(runs))}/{len(runs)} runs, {len(out)} records", flush=True)
    return pd.DataFrame(out, columns=FIELDS.split(","))


def is_zero_padding(old, new):
    return new.startswith(old) and set(new[len(old) :]) <= {"0"} and "." in old


def resolve(table, records):
    t = table.astype({"genome_id": str, "sra": str})
    r = records.dropna(subset=["sra_accession"]).copy()
    r["genome_id"] = r.genome_id.astype(str)
    r = r.assign(run=r.sra_accession.astype(str).str.split(r"[,;\s]+")).explode("run")
    r = r[r.run.ne("")].drop_duplicates(["genome_id", "run"])
    m = t[["strain_index", "genome_id", "sra", "size", "contigs"]].merge(
        r[["run", "genome_id", "genome_length", "contigs"]].rename(
            columns={"genome_id": "bvbrc_id", "contigs": "bvbrc_contigs"}
        ),
        left_on="sra",
        right_on="run",
        how="left",
    )
    ok = m[
        (pd.to_numeric(m.genome_length) == pd.to_numeric(m["size"]))
        & (pd.to_numeric(m.bvbrc_contigs) == pd.to_numeric(m.contigs))
    ]
    counts = ok.groupby("strain_index").bvbrc_id.nunique().reindex(t.strain_index, fill_value=0)
    zero = counts.index[counts.eq(0)].tolist()
    many = counts.index[counts.gt(1)].tolist()
    if zero or many:
        raise ValueError(f"{len(zero)} strains with no match {zero[:10]}, {len(many)} with several {many[:10]}")
    new = ok.drop_duplicates("strain_index").set_index("strain_index").bvbrc_id.reindex(t.strain_index)
    old = t.set_index("strain_index").genome_id
    bad = [i for i in old.index if not is_zero_padding(old[i], new[i])]
    if bad:
        raise ValueError(
            f"{len(bad)} IDs change by more than appended zeros, e.g. {[(old[i], new[i]) for i in bad[:5]]}"
        )
    if new.duplicated().any():
        raise ValueError(f"resolved IDs not unique: {new[new.duplicated()].tolist()[:10]}")
    fixed = t.copy()
    fixed["genome_id"] = new.values
    return fixed, int((old.values != new.values).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", default=str(TABLE))
    ap.add_argument("--records", default=None)
    ap.add_argument("--save-records", default=None)
    a = ap.parse_args()
    table = pd.read_csv(a.table, dtype=str)
    if a.records:
        records = pd.read_csv(a.records, dtype=str)
    else:
        records = fetch_records(table.sra.tolist())
        if a.save_records:
            records.to_csv(a.save_records, index=False)
    fixed, changed = resolve(table, records)
    fixed.to_csv(a.table, index=False)
    print(json.dumps({"strains": len(fixed), "ids_changed": changed}))


if __name__ == "__main__":
    main()
