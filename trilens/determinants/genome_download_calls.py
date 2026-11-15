from __future__ import annotations

import argparse
import gzip
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

from trilens.determinants.amrfinder_runner import MIN_COVERAGE, MIN_IDENTITY, parse_hits

ROOT = Path(__file__).resolve().parents[2]
URLS = [
    "https://www.bv-brc.org/api/genome_sequence/?eq(genome_id,{g})&limit(25000)",
    "https://ftp.bvbrc.org/genomes/{g}/{g}.fna",
]
HEADERS = {"Accept": "application/dna+fasta", "User-Agent": "trilens-amr"}
MIN_STRAINS = 5
MAX_FAILS = 2

STATUS_COLUMNS = ["strain_index", "genome_id", "time", "status", "scheme", "mb", "download_s", "amrfinder_s", "error"]


def manifest():
    repo = os.environ.get("TRILENS_REPO_RESULTS")
    p = (Path(repo) if repo else ROOT / "results") / "data_inventory" / "genome_ids.csv"
    return pd.read_csv(p, dtype={"genome_id": str})


def db_args(db):
    return ["--database", str(db)] if db else []


def versions(db=None):
    v = subprocess.run(["amrfinder", "--version"], capture_output=True, text=True).stdout.strip()
    db = subprocess.run(["amrfinder", "--database_version", *db_args(db)], capture_output=True, text=True).stdout
    db = next((line.split(":", 1)[1].strip() for line in db.splitlines() if "atabase version" in line), db.strip())
    return {"software": v, "database": db}


def download(gid, dest, tries=3):
    last = None
    for attempt in range(tries):
        for tmpl in URLS:
            try:
                req = urllib.request.Request(tmpl.format(g=gid), headers=HEADERS)
                with urllib.request.urlopen(req, timeout=300) as r, open(dest, "wb") as f:
                    shutil.copyfileobj(r, f)
                if dest.stat().st_size > 1_000_000:
                    return "api" if "/api/" in tmpl else tmpl.split(":")[0]
                last = f"too small ({dest.stat().st_size} bytes)"
            except Exception as e:
                last = f"{type(e).__name__}: {getattr(e, 'code', '') or e}"[:200]
        time.sleep(5 * (attempt + 1))
    raise RuntimeError(last)


def fasta_stats(path):
    contigs = bp = 0
    with open(path, "rb") as f:
        for line in f:
            if line.startswith(b">"):
                contigs += 1
            else:
                bp += len(line.strip())
    return contigs, bp


def one(gid, out_tsv, tmp, threads, db=None, expect=None):
    fna = tmp / f"{gid}.fna"
    t0 = time.time()
    try:
        scheme = download(gid, fna)
        t1 = time.time()
        if expect is not None and fasta_stats(fna) != expect:
            raise RuntimeError(f"assembly differs: got {fasta_stats(fna)} (contigs, bp), expected {expect}")
        raw = tmp / f"{gid}.tsv"
        subprocess.run(
            [
                "amrfinder",
                "-n",
                str(fna),
                "--organism",
                "Escherichia",
                "--plus",
                "--threads",
                str(threads),
                *db_args(db),
                "-o",
                str(raw),
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )
        part = out_tsv.with_name(out_tsv.name + ".part")
        with open(raw, "rb") as fi, gzip.open(part, "wb") as fo:
            shutil.copyfileobj(fi, fo)
        os.replace(part, out_tsv)
        raw.unlink()
        return dict(
            status="ok",
            scheme=scheme,
            mb=round(fna.stat().st_size / 1e6, 2),
            download_s=round(t1 - t0, 1),
            amrfinder_s=round(time.time() - t1, 1),
            error="",
        )
    except subprocess.CalledProcessError as e:
        return dict(status="amrfinder_failed", error=(e.stderr or "")[-200:])
    except Exception as e:
        return dict(status="download_failed", error=str(e)[:200])
    finally:
        fna.unlink(missing_ok=True)


def use_pinned(a):
    root = getattr(a, "amrfinder_root", None)
    if not root:
        return
    from trilens.determinants.amrfinder_install import DATABASE

    os.environ["PATH"] = f"{Path(root) / 'env' / 'bin'}{os.pathsep}{os.environ['PATH']}"
    if not getattr(a, "database", None):
        a.database = str(Path(root) / "db" / DATABASE)


def fetch(a):
    use_pinned(a)
    out = Path(a.out) / f"shard_{a.shard}"
    tsv_dir = out / "tsv"
    tsv_dir.mkdir(parents=True, exist_ok=True)
    tmp = Path("/tmp/arg_fetch")
    tmp.mkdir(exist_ok=True)
    status_p = out / "status.csv"
    m = manifest()
    mine = m[m.strain_index % a.of == a.shard]
    v = versions(getattr(a, "database", None))
    vp = out / "versions.json"
    if vp.exists() and json.loads(vp.read_text()) != v:
        sys.exit(
            f"this shard was started with {vp.read_text()} but amrfinder is now {v}; "
            "do not mix versions inside one shard"
        )
    vp.write_text(json.dumps(v, indent=1))

    deadline = time.time() + a.hours * 3600
    for _pass in range(MAX_FAILS):
        past = pd.read_csv(status_p, dtype={"genome_id": str}) if status_p.exists() else None
        fails = past[past.status.ne("ok")].genome_id.value_counts() if past is not None else {}
        todo = [
            (int(r.strain_index), r.genome_id, expected(r))
            for r in mine.itertuples()
            if not (tsv_dir / f"{r.genome_id}.tsv.gz").exists() and fails.get(r.genome_id, 0) < MAX_FAILS
        ]
        print(f"shard {a.shard}/{a.of}: {len(mine)} strains, {len(todo)} to do, amrfinder {v}", flush=True)
        if not todo or time.time() >= deadline:
            break
        run_pass(todo, a, tsv_dir, tmp, status_p, deadline)
    finish(out, tsv_dir, status_p, mine, v)


def expected(r):
    c, s = getattr(r, "contigs", None), getattr(r, "size", None)
    return None if pd.isna(c) or pd.isna(s) else (int(c), int(s))


def run_pass(todo, a, tsv_dir, tmp, status_p, deadline):
    done = 0
    with ThreadPoolExecutor(a.workers) as pool:
        futs = {}
        it = iter(todo)

        def submit():
            if time.time() < deadline:
                nxt = next(it, None)
                if nxt:
                    futs[
                        pool.submit(
                            one,
                            nxt[1],
                            tsv_dir / f"{nxt[1]}.tsv.gz",
                            tmp,
                            a.threads,
                            getattr(a, "database", None),
                            nxt[2],
                        )
                    ] = nxt

        for _ in range(a.workers * 2):
            submit()
        while futs:
            fut = next(as_completed(futs))
            idx, gid, _ = futs.pop(fut)
            row = dict(strain_index=idx, genome_id=gid, time=time.strftime("%Y-%m-%d %H:%M:%S"), **fut.result())
            pd.DataFrame([row]).reindex(columns=STATUS_COLUMNS).to_csv(
                status_p, mode="a", header=not status_p.exists(), index=False
            )
            done += 1
            if done % 25 == 0:
                print(f"  {done}/{len(todo)}  last: {gid} {row['status']}", flush=True)
            submit()


def finish(out, tsv_dir, status_p, mine, v):
    st = (
        pd.read_csv(status_p, dtype={"genome_id": str})
        if status_p.exists()
        else pd.DataFrame(columns=["genome_id", "status"])
    )
    have = {p.name[: -len(".tsv.gz")] for p in tsv_dir.glob("*.tsv.gz")}
    gave_up = set(st[st.status.ne("ok")].genome_id.value_counts().loc[lambda s: s >= MAX_FAILS].index) - have
    left = [g for g in mine.genome_id if g not in have and g not in gave_up]
    print(f"{out.name}: {len(have)} ok, {len(gave_up)} given up, {len(left)} left", flush=True)
    if left:
        print("time budget spent: run this notebook again to continue", flush=True)
        sys.exit(3)
    (out / "_DONE.json").write_text(json.dumps({"ok": len(have), "given_up": sorted(gave_up), **v}, indent=1))


def build(a):
    calls = Path(a.calls)
    shards = sorted(calls.glob("shard_*"))
    if not shards:
        sys.exit(f"no shards under {calls}")
    unfinished = [s.name for s in shards if not (s / "_DONE.json").exists()]
    if unfinished and not getattr(a, "allow_partial", False):
        sys.exit(f"shards not finished: {unfinished}; run those notebooks again (or pass --allow-partial)")
    vs = {s.name: json.loads((s / "versions.json").read_text()) for s in shards}
    if len({json.dumps(v, sort_keys=True) for v in vs.values()}) != 1:
        sys.exit(f"shards used different AMRFinderPlus versions: {vs}")
    m = manifest().set_index("genome_id")
    frames = []
    for s in shards:
        for p in sorted((s / "tsv").glob("*.tsv.gz")):
            gid = p.name[: -len(".tsv.gz")]
            with gzip.open(p, "rt") as f:
                text = f.read()
            tmp = Path("/tmp/arg_build.tsv")
            tmp.write_text(text)
            h = parse_hits(tmp)
            h.insert(0, "strain_index", int(m.loc[gid, "strain_index"]))
            frames.append(h if len(h) else pd.DataFrame({"strain_index": [int(m.loc[gid, "strain_index"])]}))
    long = pd.concat(frames, ignore_index=True)
    strains = sorted(long.strain_index.unique())
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    long.to_csv(out / "hits_long.csv.gz", index=False)

    ok = long.dropna(subset=["symbol"])
    passed = ok[(ok.identity >= MIN_IDENTITY) & (ok.coverage >= MIN_COVERAGE)] if "identity" in ok else ok
    amr = passed[passed["type"].str.upper().eq("AMR")]
    acquired = amr[~amr.is_point]
    point = ok[ok.is_point & ok["type"].str.upper().eq("AMR")]

    def matrix(df, min_strains):
        x = pd.crosstab(df.strain_index, df.symbol).clip(upper=1).reindex(strains, fill_value=0)
        return x.loc[:, x.sum() >= min_strains].astype("uint8")

    A, P = matrix(acquired, MIN_STRAINS), matrix(point, MIN_STRAINS)
    A.to_csv(out / "acquired.csv")
    P.to_csv(out / "point.csv")
    n_all = len(manifest())
    cov = {
        "strains_with_table": len(strains),
        "strains_total": n_all,
        "coverage": round(len(strains) / n_all, 4),
        "acquired_columns": A.shape[1],
        "point_columns": P.shape[1],
        **next(iter(vs.values())),
        "top_acquired": A.sum().sort_values(ascending=False).head(40).astype(int).to_dict(),
    }
    (out / "coverage.json").write_text(json.dumps(cov, indent=1))
    print(json.dumps(cov, indent=1))


def summary(a):
    st = pd.concat([pd.read_csv(f, dtype={"genome_id": str}) for f in sorted(Path(a.calls).glob("shard_*/status.csv"))])
    ok = st[st.status.eq("ok")].copy()
    ok["total_s"] = ok.download_s + ok.amrfinder_s
    out = {
        "genomes_tried": int(st.genome_id.nunique()),
        "ok": int(len(ok)),
        "failed": st[st.status.ne("ok")].groupby("status").size().astype(int).to_dict(),
        "errors": st[st.status.ne("ok")].error.dropna().str[:120].value_counts().head(5).astype(int).to_dict(),
        "scheme": ok.scheme.value_counts().astype(int).to_dict(),
        "median_download_s": float(ok.download_s.median()),
        "median_amrfinder_s": float(ok.amrfinder_s.median()),
        "median_total_s": float(ok.total_s.median()),
        "p90_total_s": float(ok.total_s.quantile(0.9)),
        "median_mb": float(ok.mb.median()),
        "workers": a.workers,
        **json.loads(next(Path(a.calls).glob("shard_*/versions.json")).read_text()),
    }
    for n in (1890, 6962):
        out[f"session_hours_for_{n}"] = round(out["median_total_s"] * n / a.workers / 3600, 1)
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch")
    f.add_argument("--shard", type=int, required=True)
    f.add_argument("--of", type=int, required=True)
    f.add_argument("--out", default="results/determinant_calls")
    f.add_argument("--workers", type=int, default=2)
    f.add_argument("--threads", type=int, default=2)
    f.add_argument("--hours", type=float, default=11.0)
    f.add_argument("--database", default=None)
    f.add_argument("--amrfinder-root", default=None)
    b = sub.add_parser("build")
    b.add_argument("--calls", default="results/determinant_calls")
    b.add_argument("--out", default="results/determinant_matrix")
    b.add_argument("--allow-partial", action="store_true")
    s = sub.add_parser("summary")
    s.add_argument("--calls", default="results/determinant_calls")
    s.add_argument("--out", default="results/determinant_calls/pilot_summary.json")
    s.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()
    {"fetch": fetch, "build": build, "summary": summary}[a.cmd](a)


if __name__ == "__main__":
    main()
