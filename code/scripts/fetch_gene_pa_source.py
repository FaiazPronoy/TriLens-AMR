"""
fetch_gene_pa_source.py
=======================
Finds Arcadia's presence/absence PLINK files (presence_absence_all.bed/.bim/.fam) and
SRA_to_genome_name.csv for scripts/repair_gene_pa.py.

Looks under /kaggle/input first (the raw-download dataset, if attached). Otherwise
downloads the Arcadia release from Zenodo (record 14364732, Ecoli_AMR_GenotypePhenotype.zip,
2.4 GB, md5 checked) and extracts only those files.

    python scripts/fetch_gene_pa_source.py --dest /tmp/pa_source

Writes <dest>/paths.json with the paths it found.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import zipfile
from pathlib import Path

ZENODO_URL = "https://zenodo.org/records/14364732/files/Ecoli_AMR_GenotypePhenotype.zip?download=1"
ZENODO_MD5 = "18f9d97d884d29fb30dedea7a41652ea"
WANT = ("presence_absence_all.bed", "presence_absence_all.bim", "presence_absence_all.fam",
        "SRA_to_genome_name.csv")


def find(root: Path):
    hits = {}
    for name in WANT:
        found = sorted(root.rglob(name)) if root.exists() else []
        if found:
            hits[name] = found[0]
    return hits


def download(dest: Path) -> Path:
    import requests
    z = dest / "arcadia_release2.zip"
    if z.exists() and md5(z) == ZENODO_MD5:
        return z
    print("downloading", ZENODO_URL, flush=True)
    with requests.get(ZENODO_URL, stream=True, timeout=600) as r:
        r.raise_for_status()
        with open(z, "wb") as fh:
            for i, chunk in enumerate(r.iter_content(8 << 20)):
                fh.write(chunk)
                if i % 32 == 0:
                    print(f"  {fh.tell() / 1e9:.2f} GB", flush=True)
    got = md5(z)
    if got != ZENODO_MD5:
        raise SystemExit(f"md5 {got} != {ZENODO_MD5}: download damaged")
    return z


def md5(p: Path) -> str:
    h = hashlib.md5()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", required=True)
    ap.add_argument("--input", default="/kaggle/input")
    a = ap.parse_args()
    dest = Path(a.dest)
    dest.mkdir(parents=True, exist_ok=True)

    hits = find(Path(a.input))
    source = "kaggle input"
    if len(hits) < len(WANT):
        source = "zenodo 14364732"
        z = download(dest)
        with zipfile.ZipFile(z) as zf:
            for m in zf.infolist():
                base = Path(m.filename).name
                if base in WANT and base not in hits and "__MACOSX" not in m.filename:
                    out = dest / "src" / base
                    out.parent.mkdir(parents=True, exist_ok=True)
                    with zf.open(m) as s, open(out, "wb") as d:
                        shutil.copyfileobj(s, d)
                    hits[base] = out
        z.unlink()                                     # 2.4 GB, not needed any more
    missing = [n for n in WANT if n not in hits]
    if missing:
        raise SystemExit(f"not found: {missing}")
    # the three PLINK files must sit side by side under one stem
    bed = hits["presence_absence_all.bed"]
    for ext in (".bim", ".fam"):
        if not bed.with_suffix(ext).exists():
            raise SystemExit(f"{bed.with_suffix(ext)} missing next to the .bed")
    info = {"source": source, "bed": str(bed), "sra": str(hits["SRA_to_genome_name.csv"]),
            "bed_md5": md5(bed), "bim_md5": md5(bed.with_suffix(".bim"))}
    (dest / "paths.json").write_text(json.dumps(info, indent=1))
    print(json.dumps(info, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
