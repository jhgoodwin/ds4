#!/usr/bin/env python3
"""
Whole-repo resumable Hugging Face downloader.

Downloads EVERY file in a repo (a full snapshot), unlike download_resume.py
which handles a single file. Built on the same robustness primitives:

  - file manifest fetched from the HF API (arbitrary repo id supported)
  - per-file HTTP Range resume (.part sidecar)
  - per-file retry with exponential backoff + jitter on SSL / connection errors
  - per-file size verification against the LFS size from the API manifest
  - optional SHA256 verification against the HF LFS hash
  - atomic rename (.part -> final); re-run skips already-complete files

Usage:
  # Download the whole repo (skips files already complete)
  python3 download_repo_resume.py deepseek-ai/DeepSeek-V4-Flash-0731 --out-dir /data/models

  # Show manifest without downloading
  python3 download_repo_resume.py deepseek-ai/DeepSeek-V4-Flash-0731 --list

  # Download + verify every file against its HF LFS SHA256
  python3 download_repo_resume.py deepseek-ai/DeepSeek-V4-Flash-0731 --out-dir DIR --sha256

  # Subset via glob patterns
  python3 download_repo_resume.py <repo> --only 'model-*.safetensors'
  python3 download_repo_resume.py <repo> --skip 'encoding/*' 'inference/*'

  # Start clean
  python3 download_repo_resume.py <repo> --out-dir DIR --force

  # Install into the HF Hub cache (same layout as `hf download`, honoring
  # HF_HUB_CACHE / HF_HOME). Tools then find the model by repo id, and
  # HF_HUB_OFFLINE=1 works:
  #   <cache>/models--<owner>--<repo>/blobs/<oid>          (content-addressed)
  #   <cache>/models--<owner>--<repo>/snapshots/<rev>/<f>  -> ../../blobs/<oid>
  #   <cache>/models--<owner>--<repo>/refs/main            (revision sha)
  python3 download_repo_resume.py deepseek-ai/DeepSeek-V4-Flash-0731 --cache
"""

import os
import sys
import time
import json
import random
import argparse
import ssl
import hashlib
import fnmatch
import urllib.request
import urllib.error

BASE_URL = "https://huggingface.co"
CHUNK_SIZE = 8 * 1024 * 1024  # 8 MiB
MAX_RETRIES_DEFAULT = 50
RETRY_DELAYS = [1, 2, 4, 8, 16, 30, 60]


def get_headers(token=None):
    h = {
        "User-Agent": "ds4-repo-downloader/1.0",
        "Accept": "*/*",
    }
    if token:
        h["Authorization"] = f"Bearer {token}"
    return h


# ── Manifest ─────────────────────────────────────────────────────────

def fetch_manifest(repo, token, timeout=30):
    """Fetch repo file list from the HF API.
    Returns (manifest, revision):
      manifest: list of dicts {name, size (int|None), sha256 (str|None)}
      revision: commit sha of `main` (str)"""
    url = f"{BASE_URL}/api/models/{repo}?blobs=true"
    headers = get_headers(token)
    req = urllib.request.Request(url, headers=headers, unverifiable=True)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read())

    siblings = data.get("siblings", [])
    if not siblings:
        raise RuntimeError(f"No files found for repo {repo}")

    manifest = []
    for sib in siblings:
        lfs = sib.get("lfs", {}) or {}
        manifest.append({
            "name": sib["rfilename"],
            "size": lfs.get("size"),          # None for non-LFS (small) files
            "sha256": lfs.get("sha256"),      # None for non-LFS files
        })
    return manifest, data.get("sha")


# ── Single-file download (ported from download_resume.py) ────────────

def build_url(repo, filename):
    return f"{BASE_URL}/{repo}/resolve/main/{filename}"


def download_file(repo, entry, dest, token, max_retries):
    """Download one file with resume + retry. Returns (ok, bytes)."""
    url = build_url(repo, entry["name"])
    headers = get_headers(token)
    expected_len = entry["size"]
    part_path = dest + ".part"

    # Fully downloaded already?
    if os.path.exists(dest):
        actual = os.path.getsize(dest)
        if expected_len is None or actual == expected_len:
            print(f"  [✓] already complete: {entry['name']} ({actual / 1024**3:.2f} GiB)")
            return True, actual
        print(f"  [!] {entry['name']}: size mismatch ({actual} vs {expected_len}), re-downloading")
        os.remove(dest)

    resume_at = 0
    if os.path.exists(part_path):
        resume_at = os.path.getsize(part_path)
        if expected_len is not None and resume_at >= expected_len:
            print(f"  [*] part file already complete ({resume_at} bytes), renaming")
            os.rename(part_path, dest)
            return True, resume_at
        print(f"  [~] resuming {entry['name']} from {resume_at} bytes")

    retries = 0
    while retries <= max_retries:
        try:
            _do_download(url, dest, part_path, headers, resume_at, expected_len, entry["name"])
            final_size = os.path.getsize(dest)
            return True, final_size
        except (urllib.error.URLError, ConnectionError, OSError, ssl.SSLError, RuntimeError) as e:
            retries += 1
            if os.path.exists(part_path):
                new_size = os.path.getsize(part_path)
                if new_size > resume_at:
                    print(f"  [+] partial progress: {resume_at} -> {new_size} bytes")
                    resume_at = new_size
                elif new_size < resume_at:
                    print(f"  [!] file shrank (possible corruption), restarting")
                    os.remove(part_path)
                    resume_at = 0

            if retries > max_retries:
                print(f"  [!] exhausted {max_retries} retries on {entry['name']}", file=sys.stderr)
                return False, resume_at

            delay = RETRY_DELAYS[min(retries - 1, len(RETRY_DELAYS) - 1)] * random.uniform(0.5, 1.5)
            print(f"  [!] {e}")
            print(f"  [~] retry {retries}/{max_retries} in {delay:.1f}s...")
            time.sleep(delay)

    return False, resume_at


def _do_download(url, dest, part_path, headers, resume_at, expected_len, label):
    req_headers = dict(headers)
    if resume_at > 0:
        req_headers["Range"] = f"bytes={resume_at}-"

    req = urllib.request.Request(url, headers=req_headers, unverifiable=True)
    resp = urllib.request.urlopen(req, timeout=120)
    status = resp.status

    if status == 416:
        print(f"  [*] range not satisfiable (complete at {resume_at} bytes), renaming")
        os.rename(part_path, dest)
        return

    if status == 206:
        pass  # range honored
    elif status == 200:
        print(f"  [~] server ignored range request (200), starting from scratch")
        if os.path.exists(part_path):
            os.remove(part_path)
        resume_at = 0
    else:
        body = resp.read(512)
        raise RuntimeError(f"unexpected HTTP {status}: {body.decode(errors='replace')}")

    content_range = resp.headers.get("Content-Range")
    total = int(content_range.split("/")[-1]) if content_range else expected_len

    mode = "ab" if resume_at > 0 else "wb"
    with open(part_path, mode) as f:
        bytes_so_far = resume_at
        last_log = time.monotonic()
        while True:
            chunk = resp.read(CHUNK_SIZE)
            if not chunk:
                break
            f.write(chunk)
            bytes_so_far += len(chunk)
            now = time.monotonic()
            if now - last_log >= 30:
                _log_progress(label, bytes_so_far, total)
                last_log = now
    _log_progress(label, bytes_so_far, total)
    print()

    final_size = os.path.getsize(part_path)
    if total is not None and final_size != total:
        raise RuntimeError(
            f"size mismatch: got {final_size}, expected {total}. "
            f"Delete {part_path} and retry."
        )
    os.rename(part_path, dest)


def _log_progress(label, current, total):
    if total:
        print(f"  ... {label}: {current / 1024**3:.2f} / {total / 1024**3:.2f} GiB "
              f"({current / total * 100:.1f}%)", flush=True)
    else:
        print(f"  ... {label}: {current / 1024**3:.2f} GiB", flush=True)


# ── HF cache layout ──────────────────────────────────────────────────

def cache_root():
    """HF Hub cache root: HF_HUB_CACHE, else HF_HOME/hub, else ~/.cache/huggingface/hub."""
    env = os.environ.get("HF_HUB_CACHE")
    if env:
        return os.path.abspath(os.path.expanduser(env))
    home = os.environ.get("HF_HOME")
    if home:
        return os.path.join(os.path.abspath(os.path.expanduser(home)), "hub")
    return os.path.expanduser("~/.cache/huggingface/hub")


def head_etag(url, headers, timeout=30):
    """Resolve the blob oid for a non-LFS file via the response ETag.
    Returns (oid, None) on success, (None, error) on failure."""
    try:
        req = urllib.request.Request(url, method="HEAD", headers=headers, unverifiable=True)
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            etag = resp.headers.get("ETag") or resp.headers.get("x-linked-etag")
            if etag:
                return etag.strip('"'), None
            return None, "no ETag in HEAD response"
    except Exception as e:
        return None, str(e)


def resolve_oid(entry, url, headers):
    """Blob oid for the HF cache: LFS sha256 for LFS files, ETag otherwise."""
    if entry["sha256"]:
        return entry["sha256"], None
    return head_etag(url, headers)


def install_cache_layout(repo_cache, revision, entries, oids):
    """Create symlinks snapshots/<rev>/<name> -> ../../blobs/<oid> and write refs/main."""
    refs_dir = os.path.join(repo_cache, "refs")
    snap_dir = os.path.join(repo_cache, "snapshots", revision)
    os.makedirs(refs_dir, exist_ok=True)
    os.makedirs(snap_dir, exist_ok=True)

    with open(os.path.join(refs_dir, "main"), "w") as f:
        # huggingface_hub reads this file raw (no strip) — must have no trailing newline
        f.write(revision)

    for entry, oid in zip(entries, oids):
        if not oid:
            continue
        link = os.path.join(snap_dir, entry["name"])
        os.makedirs(os.path.dirname(link), exist_ok=True)
        target = os.path.relpath(os.path.join(repo_cache, "blobs", oid),
                                 os.path.dirname(link))
        if os.path.islink(link) and os.readlink(link) == target:
            continue
        if os.path.lexists(link):
            os.remove(link)
        os.symlink(target, link)


# ── SHA256 verification ──────────────────────────────────────────────

def compute_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(64 * 1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


# ── Main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Whole-repo resumable HF downloader (all files in a repo)")
    parser.add_argument("repo", help="Hugging Face repo id (e.g. deepseek-ai/DeepSeek-V4-Flash-0731)")
    parser.add_argument("--out-dir", default=".", help="Output directory (default: .)")
    parser.add_argument("--token", help="HF token (default: ~/.cache/huggingface/token)")
    parser.add_argument("--max-retries", type=int, default=MAX_RETRIES_DEFAULT,
                        help=f"Max retries per file on SSL/connection errors (default: {MAX_RETRIES_DEFAULT})")
    parser.add_argument("--sha256", action="store_true",
                        help="Verify each file's SHA256 against HF LFS hash (~1min/GiB)")
    parser.add_argument("--force", action="store_true",
                        help="Delete existing dest/.part files and restart")
    parser.add_argument("--list", action="store_true",
                        help="Print the file manifest and exit (no download)")
    parser.add_argument("--only", nargs="+", metavar="GLOB",
                        help="Only download files matching these glob patterns (fnmatch)")
    parser.add_argument("--skip", nargs="+", metavar="GLOB",
                        help="Skip files matching these glob patterns (fnmatch)")
    parser.add_argument("--cache", action="store_true",
                        help="Install into the HF Hub cache (blobs + snapshots + refs), "
                             "same layout as `hf download`. Honors HF_HUB_CACHE / HF_HOME. "
                             "Ignores --out-dir.")
    args = parser.parse_args()

    token = args.token
    if not token:
        token_path = os.path.expanduser("~/.cache/huggingface/token")
        if os.path.isfile(token_path):
            with open(token_path) as f:
                token = f.read().strip()

    print(f"Fetching manifest for {args.repo} ...")
    manifest, revision = fetch_manifest(args.repo, token)

    if args.only:
        manifest = [e for e in manifest
                    if any(fnmatch.fnmatch(e["name"], p) for p in args.only)]
    if args.skip:
        manifest = [e for e in manifest
                    if not any(fnmatch.fnmatch(e["name"], p) for p in args.skip)]

    if not manifest:
        print("No files match the given filters.")
        sys.exit(1)

    total = sum(e["size"] or 0 for e in manifest)
    print(f"{len(manifest)} files, {total / 1024**3:.2f} GiB total")
    if args.list:
        for e in manifest:
            sz = e["size"]
            print(f"  {e['name']:70s} {sz / 1e9:9.2f} GB" if sz else f"  {e['name']:70s}       -")
        return

    if args.cache:
        root = cache_root()
        repo_cache = os.path.join(root, f"models--{args.repo.replace('/', '--')}")
        blobs_dir = os.path.join(repo_cache, "blobs")
        os.makedirs(blobs_dir, exist_ok=True)
        print(f"Cache: {repo_cache}")
        print(f"Revision: {revision}")
    else:
        out_dir = os.path.abspath(args.out_dir)
        os.makedirs(out_dir, exist_ok=True)
        print(f"Output: {out_dir}")
    print()

    # Resolve blob oids up front (LFS sha256 from API; ETag for small files).
    oids = []
    for e in manifest:
        if args.cache:
            url = build_url(args.repo, e["name"])
            oid, err = resolve_oid(e, url, get_headers(token))
            if err:
                print(f"  [!] {e['name']}: cannot resolve cache oid ({err}), skipping", file=sys.stderr)
            oids.append(oid)
        else:
            oids.append(None)

    if args.force:
        for e, oid in zip(manifest, oids):
            if args.cache:
                if oid:
                    for p in (os.path.join(blobs_dir, oid),
                              os.path.join(blobs_dir, oid) + ".part"):
                        if os.path.exists(p):
                            os.remove(p)
            else:
                for p in (os.path.join(out_dir, e["name"]),
                          os.path.join(out_dir, e["name"]) + ".part"):
                    if os.path.exists(p):
                        os.remove(p)
        print("  [*] removed existing files (--force)")

    failed = []
    done_bytes = 0
    for e, oid in zip(manifest, oids):
        if args.cache:
            if not oid:
                continue
            dest = os.path.join(blobs_dir, oid)
        else:
            dest = os.path.join(out_dir, e["name"])
            os.makedirs(os.path.dirname(dest), exist_ok=True)

        ok, size = download_file(args.repo, e, dest, token, args.max_retries)
        if ok:
            done_bytes += size
            if args.sha256 and e["sha256"]:
                print(f"  [~] verifying sha256 of {e['name']} ...", flush=True)
                digest = compute_sha256(dest)
                if digest != e["sha256"]:
                    print(f"  [✗] SHA256 MISMATCH on {e['name']} — delete and re-run", file=sys.stderr)
                    failed.append(e["name"])
                    continue
                print(f"  [✓] sha256 matches: {e['name']}")
        else:
            failed.append(e["name"])

        print()

    if args.cache and not failed:
        install_cache_layout(repo_cache, revision, manifest, oids)

    print("=" * 60)
    print(f"Files complete: {len(manifest) - len(failed)}/{len(manifest)} "
          f"({done_bytes / 1024**3:.2f} GiB)")
    if args.cache and not failed:
        print(f"Cache ready: {repo_cache}")
        print("Tools can now find the model by repo id; HF_HUB_OFFLINE=1 works.")
    if failed:
        print(f"FAILED ({len(failed)}):")
        for f in failed:
            print(f"  {f}")
        print("Re-run the same command to resume — completed files are skipped.")
        sys.exit(1)
    print("All files downloaded.")


if __name__ == "__main__":
    main()
