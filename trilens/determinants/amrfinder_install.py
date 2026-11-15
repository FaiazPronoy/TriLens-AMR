from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import subprocess
import sys
import urllib.request
from pathlib import Path

SOFTWARE = "4.2.7"
DATABASE = "2026-03-24.1"
MAMBA = "2.9.0-0"
MAMBA_URL = f"https://github.com/mamba-org/micromamba-releases/releases/download/{MAMBA}/micromamba-linux-64"
DB_URL = "https://ftp.ncbi.nlm.nih.gov/pathogen/Antimicrobial_resistance/AMRFinderPlus/database/{major}/{db}/"
HEADERS = {"User-Agent": "trilens-amr"}


def get(url, timeout=600):
    with urllib.request.urlopen(urllib.request.Request(url, headers=HEADERS), timeout=timeout) as r:
        return r.read()


def db_files(index_html):
    return sorted({h for h in re.findall(r'href="([^"/?]+)"', index_html) if not h.startswith(("http", "#"))})


def check_versions(env, db):
    amr = Path(env) / "bin" / "amrfinder"
    sw = subprocess.run([str(amr), "--version"], capture_output=True, text=True).stdout.strip()
    out = subprocess.run([str(amr), "--database_version", "--database", str(db)], capture_output=True, text=True)
    m = re.search(r"atabase version:\s*(\S+)", out.stdout + out.stderr)
    found = {"software": sw, "database": m.group(1) if m else None}
    if found != {"software": SOFTWARE, "database": DATABASE}:
        sys.exit(f"pinned {SOFTWARE} / {DATABASE} but found {found}\n{out.stdout}{out.stderr}")
    return found


def install(a):
    root = Path(a.root)
    env, db = root / "env", root / "db" / DATABASE
    root.mkdir(parents=True, exist_ok=True)
    mamba = root / "micromamba"
    if not mamba.exists():
        blob = get(MAMBA_URL)
        want = get(MAMBA_URL + ".sha256").decode().split()[0]
        if hashlib.sha256(blob).hexdigest() != want:
            sys.exit("micromamba checksum mismatch")
        mamba.write_bytes(blob)
        mamba.chmod(mamba.stat().st_mode | stat.S_IEXEC)
    if not (env / "bin" / "amrfinder").exists():
        subprocess.run(
            [
                str(mamba),
                "create",
                "-y",
                "-q",
                "-r",
                str(root / "mamba"),
                "-p",
                str(env),
                "-c",
                "conda-forge",
                "-c",
                "bioconda",
                "--strict-channel-priority",
                f"ncbi-amrfinderplus={SOFTWARE}",
            ],
            check=True,
        )
    if not (db / "_INDEXED").exists():
        db.mkdir(parents=True, exist_ok=True)
        base = DB_URL.format(major=".".join(SOFTWARE.split(".")[:2]), db=DATABASE)
        for name in db_files(get(base).decode()):
            (db / name).write_bytes(get(base + name))
        subprocess.run([str(env / "bin" / "amrfinder_index"), str(db)], check=True)
        (db / "_INDEXED").write_text(DATABASE)
    found = check_versions(env, db)
    info = {**found, "env": str(env), "database_dir": str(db), "micromamba": MAMBA}
    Path(a.info).parent.mkdir(parents=True, exist_ok=True)
    Path(a.info).write_text(json.dumps(info, indent=1))
    print(json.dumps(info, indent=1))
    print(f"add to PATH: {env / 'bin'}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/tmp/amrfinder")
    ap.add_argument("--info", default="results/determinant_calls/amrfinder_install.json")
    install(ap.parse_args())


if __name__ == "__main__":
    os.umask(0o022)
    main()
