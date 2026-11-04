from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

API = os.environ.get("GITHUB_API", "https://api.github.com")
TEXT_SUFFIXES = {".csv", ".md", ".json", ".yaml", ".yml", ".txt", ".tsv"}
TEXT_MAX_BYTES = 5_000_000


def _session(token):
    import requests

    s = requests.Session()
    s.headers.update({"Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28"})
    if token:
        s.headers["Authorization"] = f"Bearer {token}"
    return s


def _redact(text, token):
    return text.replace(token, "***") if token else text


def _release(s, repo, tag, create):
    r = s.get(f"{API}/repos/{repo}/releases/tags/{tag}", timeout=60)
    if r.status_code == 200:
        return r.json()
    if r.status_code == 404 and create:
        r = s.post(
            f"{API}/repos/{repo}/releases",
            timeout=60,
            json={
                "tag_name": tag,
                "name": tag,
                "prerelease": True,
                "body": "Outputs of the Kaggle notebook of the same name. Written by trilens/utils/github_sync.py.",
            },
        )
        if r.status_code in (200, 201):
            return r.json()
    raise RuntimeError(f"release {tag}: HTTP {r.status_code} {r.text[:200]}")


def upload_asset(repo, token, tag, path):
    path = Path(path)
    try:
        s = _session(token)
        rel = _release(s, repo, tag, create=True)
        part = path.name + ".part"
        for a in rel.get("assets", []):
            if a["name"] == part:
                s.delete(f"{API}/repos/{repo}/releases/assets/{a['id']}", timeout=60)
        url = rel["upload_url"].split("{")[0]
        with open(path, "rb") as fh:
            r = s.post(url, params={"name": part}, data=fh, timeout=3600, headers={"Content-Type": "application/zip"})
        if r.status_code not in (200, 201):
            raise RuntimeError(f"HTTP {r.status_code} {r.text[:200]}")
        new_id = r.json()["id"]
        for a in rel.get("assets", []):
            if a["name"] == path.name:
                s.delete(f"{API}/repos/{repo}/releases/assets/{a['id']}", timeout=60)
        r = s.patch(f"{API}/repos/{repo}/releases/assets/{new_id}", json={"name": path.name}, timeout=60)
        if r.status_code != 200:
            raise RuntimeError(f"rename: HTTP {r.status_code} {r.text[:200]}")
        print(f"[github] {path.name} -> release {tag} ({path.stat().st_size / 1e6:.1f} MB)")
        return True
    except Exception as exc:
        print(f"[github] upload of {path.name} failed: {_redact(str(exc), token)}")
        return False


def download_asset(repo, token, tag, name, dest_dir):
    try:
        s = _session(token)
        rel = _release(s, repo, tag, create=False)
        asset = next((a for a in rel.get("assets", []) if a["name"] == name), None)
        if asset is None:
            return None
        dest = Path(dest_dir) / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        with s.get(asset["url"], headers={"Accept": "application/octet-stream"}, stream=True, timeout=3600) as r:
            r.raise_for_status()
            with open(dest, "wb") as fh:
                for chunk in r.iter_content(1 << 20):
                    fh.write(chunk)
        print(f"[github] release {tag}: {name} ({dest.stat().st_size / 1e6:.1f} MB)")
        return dest
    except Exception as exc:
        print(f"[github] no {name} from release {tag}: {_redact(str(exc), token)}")
        return None


def push_text_results(remote_url, branch, results_dir, prefix="", message="results", token=None):
    results_dir = Path(results_dir)
    tmp = Path(tempfile.mkdtemp(prefix="results_branch_"))
    try:
        n = 0
        for f in sorted(results_dir.rglob("*")):
            if not f.is_file() or f.suffix.lower() not in TEXT_SUFFIXES:
                continue
            rel = f.relative_to(results_dir)
            if {"cache", "ckpts", "__pycache__"} & set(rel.parts) or f.stat().st_size > TEXT_MAX_BYTES:
                continue
            dst = tmp / prefix / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, dst)
            n += 1
        if n == 0:
            return False

        def git(*a):
            return subprocess.run(["git", *a], cwd=tmp, check=True, capture_output=True, text=True)

        git("init", "-q")
        git("checkout", "-q", "-b", branch)
        git("add", "-A")
        git(
            "-c",
            "user.name=trilens-kaggle",
            "-c",
            "user.email=trilens-kaggle@users.noreply.github.com",
            "commit",
            "-q",
            "-m",
            message,
        )
        auth = []
        if token:
            import base64

            auth = [
                "-c",
                "http.extraheader=AUTHORIZATION: basic "
                + base64.b64encode(f"x-access-token:{token}".encode()).decode(),
            ]
        git(*auth, "push", "-q", "--force", remote_url, f"{branch}:{branch}")
        print(f"[github] {n} text files -> branch {branch}")
        return True
    except subprocess.CalledProcessError as exc:
        print(f"[github] push to {branch} failed: {_redact((exc.stderr or '') + (exc.stdout or ''), token)[:300]}")
        return False
    except Exception as exc:
        print(f"[github] push to {branch} failed: {_redact(str(exc), token)}")
        return False
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def remote(repo, token=None):
    return f"https://github.com/{repo}.git"
