#!/usr/bin/env python3
"""
Robust resumable GGUF downloader with integrity verification.

Downloads from Hugging Face with:
  - HTTP Range resume (no restart on crash)
  - Exponential backoff on SSL / connection errors
  - GGUF structural validation (all tensor offsets within file)
  - Optional ds4 --inspect integration
  - Atomic rename (.part -> final)

Usage:
  python3 download_resume.py <repo> <filename> [--out-dir DIR] [--token TOKEN]
                             [--force] [--verify] [--sha256] [--max-retries N]

Examples:
  # Download with GGUF structural verification (instant)
  python3 download_resume.py antirez/deepseek-v4-gguf \
      DeepSeek-V4-Flash-Q4KExperts-F16HC-F16Compressor-F16Indexer-Q8Attn-Q8Shared-Q8Out-chat-v2-imatrix.gguf \
      --out-dir ./gguf --verify

  # Download with full SHA256 verification against HF LFS hash (~1min/GiB)
  python3 download_resume.py antirez/deepseek-v4-gguf \
      DeepSeek-V4-Flash-Q4KExperts-F16HC-F16Compressor-F16Indexer-Q8Attn-Q8Shared-Q8Out-chat-v2-imatrix.gguf \
      --out-dir ./gguf --sha256

  # Verify GGUF structure of an existing file (instant)
  python3 download_resume.py antirez/deepseek-v4-gguf <filename> --verify-only

  # Full SHA256 of an existing file against HF LFS hash
  python3 download_resume.py antirez/deepseek-v4-gguf <filename> --sha256-only
"""

import os
import sys
import time
import random
import argparse
import ssl
import struct
import json
import subprocess
import hashlib
import urllib.request
import urllib.error

BASE_URL = "https://huggingface.co"
CHUNK_SIZE = 8 * 1024 * 1024  # 8 MiB
MAX_RETRIES_DEFAULT = 50
RETRY_DELAYS = [1, 2, 4, 8, 16, 30, 60]

# GGUF tensor type sizes from ds4.c's gguf_types[] table.
# {type_code: (name, block_elems, block_bytes)}
GGUF_TYPES = {
    0:  ("f32",      1,   4),
    1:  ("f16",      1,   2),
    2:  ("q4_0",    32,  18),
    3:  ("q4_1",    32,  20),
    6:  ("q5_0",    32,  22),
    7:  ("q5_1",    32,  24),
    8:  ("q8_0",    32,  34),
    9:  ("q8_1",    32,  40),
    10: ("q2_k",   256,  84),
    11: ("q3_k",   256, 110),
    12: ("q4_k",   256, 144),
    13: ("q5_k",   256, 176),
    14: ("q6_k",   256, 210),
    15: ("q8_k",   256, 292),
    16: ("iq2_xxs",256,  66),
    17: ("iq2_xs", 256,  74),
    18: ("iq3_xxs",256,  98),
    19: ("iq1_s",  256, 110),
    20: ("iq4_nl", 256,  50),
    21: ("iq3_s",  256, 110),
    22: ("iq2_s",  256,  82),
    23: ("iq4_xs", 256, 136),
    24: ("i8",       1,   1),
    25: ("i16",      1,   2),
    26: ("i32",      1,   4),
    27: ("i64",      1,   8),
    28: ("f64",      1,   8),
    29: ("iq1_m",   256,  56),
    30: ("bf16",     1,   2),
}


def build_url(repo, filename):
    return f"{BASE_URL}/{repo}/resolve/main/{filename}"


def get_headers(token=None):
    h = {
        "User-Agent": "ds4-downloader/1.0",
        "Accept": "*/*",
    }
    if token:
        h["Authorization"] = f"Bearer {token}"
    return h


def expected_size(url, headers, timeout=30):
    req = urllib.request.Request(url, method="HEAD", headers=headers, unverifiable=True)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            length = resp.headers.get("Content-Length")
            if length is not None:
                return int(length)
    except Exception as e:
        print(f"  [!] HEAD request failed: {e}", file=sys.stderr)
    return None


def get_lfs_sha256(repo, filename, token, timeout=30):
    """Fetch the LFS SHA256 from the Hugging Face API.
    Returns hex string or None if unavailable."""
    url = f"{BASE_URL}/api/models/{repo}?blobs=true"
    h = {
        "User-Agent": "ds4-downloader/1.0",
        "Authorization": f"Bearer {token}",
    } if token else {"User-Agent": "ds4-downloader/1.0"}
    try:
        req = urllib.request.Request(url, headers=h, unverifiable=True)
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read())
        for sib in data.get("siblings", []):
            if sib.get("rfilename") == filename:
                lfs = sib.get("lfs", {})
                if lfs:
                    return lfs.get("sha256")
        return None
    except Exception as e:
        print(f"  [!] LFS SHA256 fetch failed: {e}", file=sys.stderr)
        return None


def compute_sha256(path, progress_cb=None):
    """Compute SHA256 of a file. Returns hex digest string.
    progress_cb(bytes_processed) is called periodically if provided."""
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(64 * 1024 * 1024)  # 64 MiB for speed
            if not chunk:
                break
            h.update(chunk)
            if progress_cb:
                progress_cb(f.tell())
    return h.hexdigest()


# ── GGUF structural verification ──────────────────────────────────────

def verify_gguf(path):
    """
    Parse GGUF header and validate every tensor's data fits within the file.
    Returns (ok, summary_str).
    """
    if not os.path.exists(path):
        return False, f"File not found: {path}"

    size = os.path.getsize(path)
    if size < 12:
        return False, f"File too small ({size} bytes), not a valid GGUF"

    try:
        with open(path, "rb") as f:
            magic = f.read(4)
            if magic != b"GGUF":
                return False, f"Bad magic: {magic.hex()} (expected 47475546)"

            version = struct.unpack("<I", f.read(4))[0]
            tensor_count = struct.unpack("<Q", f.read(8))[0]
            metadata_kv_count = struct.unpack("<Q", f.read(8))[0]

            # Skip metadata KVs (we don't need their values for integrity)
            # Value types: 0=uint8, 1=int8, 2=uint16, 3=int16, 4=uint32, 5=int32,
            # 6=float32, 7=bool, 8=string, 9=array, 10=uint64, 11=int64,
            # 12=float64, 13=int16, 14=uint8 (GGUFv3 back-compat)
            value_size_lookup = {
                0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4,
                7: 1, 10: 8, 11: 8, 12: 8, 13: 2, 14: 1, 15: 1,
            }
            for _ in range(metadata_kv_count):
                klen = struct.unpack("<Q", f.read(8))[0]
                f.read(klen)  # key string
                vtype = struct.unpack("<I", f.read(4))[0]
                if vtype == 8:  # string
                    slen = struct.unpack("<Q", f.read(8))[0]
                    f.read(slen)
                elif vtype == 9:  # array
                    arr_type = struct.unpack("<I", f.read(4))[0]
                    arr_len = struct.unpack("<Q", f.read(8))[0]
                    elem_size = value_size_lookup.get(arr_type, 4)
                    if arr_type == 8:
                        for _ in range(arr_len):
                            sl = struct.unpack("<Q", f.read(8))[0]
                            f.read(sl)
                    else:
                        f.read(arr_len * elem_size)
                else:
                    vsize = value_size_lookup.get(vtype, 4)
                    f.read(vsize)

            header_end = f.tell()

            # Read tensor info
            max_data_end = 0
            type_counts = {}
            logical_params = 0

            for i in range(tensor_count):
                nlen = struct.unpack("<Q", f.read(8))[0]
                f.read(nlen)  # tensor name

                n_dims = struct.unpack("<I", f.read(4))[0]
                dims = [struct.unpack("<Q", f.read(8))[0] for _ in range(n_dims)]
                ttype = struct.unpack("<I", f.read(4))[0]
                offset = struct.unpack("<Q", f.read(8))[0]

                ti = GGUF_TYPES.get(ttype)
                if ti is None:
                    return False, f"Unknown tensor type {ttype} at tensor {i}"

                name, block_elems, block_bytes = ti
                total_elems = 1
                for d in dims:
                    total_elems *= d
                blocks = (total_elems + block_elems - 1) // block_elems
                tensor_bytes = blocks * block_bytes
                tensor_end = offset + tensor_bytes

                if tensor_end > max_data_end:
                    max_data_end = tensor_end

                type_counts[ttype] = type_counts.get(ttype, 0) + 1
                logical_params += total_elems

            tensor_info_end = f.tell()

        # Build summary
        ok = max_data_end <= size
        gap = size - max_data_end

        lines = []
        lines.append(f"GGUF v{version}, {metadata_kv_count} metadata keys, {tensor_count} tensors")
        lines.append(f"Header + tensor info: {tensor_info_end} bytes")
        lines.append(f"Tensor data end:     {max_data_end} bytes ({max_data_end / 1024**3:.2f} GiB)")
        lines.append(f"File size:           {size} bytes ({size / 1024**3:.2f} GiB)")

        if ok:
            lines.append(f"Padding / alignment: {gap} bytes")
        else:
            lines.append(f"OVERFLOW:            {-gap} bytes beyond file — CORRUPT")

        # Group by type
        lines.append("")

        # Re-scan for per-type byte totals
        type_bytes = {}
        total_data = 0
        with open(path, "rb") as f:
            f.seek(header_end)
            for i in range(tensor_count):
                nlen = struct.unpack("<Q", f.read(8))[0]
                f.read(nlen)
                n_dims = struct.unpack("<I", f.read(4))[0]
                dims = [struct.unpack("<Q", f.read(8))[0] for _ in range(n_dims)]
                ttype = struct.unpack("<I", f.read(4))[0]
                offset = struct.unpack("<Q", f.read(8))[0]
                ti = GGUF_TYPES.get(ttype)
                if ti:
                    _, block_elems, block_bytes = ti
                    total_elems = 1
                    for d in dims:
                        total_elems *= d
                    blocks = (total_elems + block_elems - 1) // block_elems
                    tensor_bytes = blocks * block_bytes
                    type_bytes[ttype] = type_bytes.get(ttype, 0) + tensor_bytes
                    total_data += tensor_bytes

        lines.append(f"{'Type':<12} {'Count':>6} {'Bytes':>14} {'GiB':>8}")
        lines.append("-" * 42)
        for ttype in sorted(type_bytes.keys()):
            ti = GGUF_TYPES.get(ttype)
            if ti is None:
                continue
            name, _, _ = ti
            count = type_counts[ttype]
            tb = type_bytes[ttype]
            lines.append(f"{name:<12} {count:>6} {tb:>14} {tb/1024**3:>7.2f}")
        lines.append("-" * 42)
        lines.append(f"{'total':<12} {tensor_count:>6} {total_data:>14} {total_data/1024**3:>7.2f}")
        lines.append(f"logical parameters: {logical_params}")

        verdict = "✓ All tensor data within file bounds" if ok else "✗ DATA BEYOND FILE BOUNDS — CORRUPT"
        lines.append("")
        lines.append(verdict)

        return ok, "\n".join(lines)

    except Exception as e:
        return False, f"GGUF parse error: {e}"


def try_ds4_inspect(path):
    """Run ds4 --inspect if available."""
    # Look for ds4 binary relative to this script or in PATH
    script_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(script_dir, "ds4"),
        os.path.join(script_dir, "ds4"),
        "ds4",
    ]
    ds4_bin = None
    for c in candidates:
        try:
            subprocess.run([c, "--help"], capture_output=True, timeout=5)
            ds4_bin = c
            break
        except (FileNotFoundError, PermissionError, subprocess.TimeoutExpired):
            continue

    if not ds4_bin:
        return None

    try:
        result = subprocess.run(
            [ds4_bin, "-m", path, "--inspect"],
            capture_output=True, timeout=120, text=True,
        )
        if result.returncode == 0:
            return result.stdout + result.stderr
        return f"(ds4 --inspect exited {result.returncode})"
    except subprocess.TimeoutExpired:
        return "(ds4 --inspect timed out after 120s)"
    except Exception as e:
        return f"(ds4 --inspect error: {e})"


# ── Download logic ────────────────────────────────────────────────────

def run_download(url, dest, part_path, headers, expected_len, max_retries):
    resume_at = 0
    if os.path.exists(part_path):
        resume_at = os.path.getsize(part_path)
        if expected_len is not None and resume_at >= expected_len:
            print(f"  [*] Part file already complete ({resume_at} bytes). Renaming...")
            os.rename(part_path, dest)
            return True
        print(f"  [*] Resuming from {resume_at} bytes ({part_path})")
    else:
        print(f"  [*] Starting fresh download")

    retries = 0
    while retries <= max_retries:
        try:
            _do_download(url, dest, part_path, headers, resume_at, expected_len)
            return True
        except (urllib.error.URLError, ConnectionError, OSError, ssl.SSLError) as e:
            retries += 1
            if os.path.exists(part_path):
                new_size = os.path.getsize(part_path)
                if new_size > resume_at:
                    print(f"  [+] Partial progress: {resume_at} -> {new_size} bytes")
                    resume_at = new_size
                elif new_size < resume_at:
                    print(f"  [!] File shrank (possible corruption). Restarting.")
                    os.remove(part_path)
                    resume_at = 0

            if retries > max_retries:
                print(f"  [!] Exhausted {max_retries} retries. Giving up.", file=sys.stderr)
                return False

            delay = RETRY_DELAYS[min(retries - 1, len(RETRY_DELAYS) - 1)]
            jitter = random.uniform(0.5, 1.5)
            actual_delay = delay * jitter
            print(f"  [!] Error: {e}")
            print(f"  [~] Retry {retries}/{max_retries} in {actual_delay:.1f}s...")
            time.sleep(actual_delay)

    return False


def _do_download(url, dest, part_path, headers, resume_at, expected_len):
    req_headers = dict(headers)
    if resume_at > 0:
        req_headers["Range"] = f"bytes={resume_at}-"

    req = urllib.request.Request(url, headers=req_headers, unverifiable=True)
    resp = urllib.request.urlopen(req, timeout=120)

    status = resp.status
    if status == 416:
        print(f"  [*] Server says range not satisfiable (file complete at {resume_at} bytes)")
        os.rename(part_path, dest)
        return

    if status == 206:
        print(f"  [+] Server accepted range request (206 Partial Content)")
    elif status == 200:
        print(f"  [~] Server ignored range request (200 OK). Starting from scratch.")
        if os.path.exists(part_path):
            os.remove(part_path)
        resume_at = 0
    else:
        resp_body = resp.read(512)
        raise RuntimeError(f"Unexpected HTTP {status}: {resp_body.decode(errors='replace')}")

    content_range = resp.headers.get("Content-Range")
    if content_range:
        total = int(content_range.split("/")[-1])
    else:
        total = expected_len

    mode = "ab" if resume_at > 0 else "wb"
    with open(part_path, mode) as f:
        bytes_this_attempt = resume_at
        last_log = time.monotonic()

        while True:
            chunk = resp.read(CHUNK_SIZE)
            if not chunk:
                break
            f.write(chunk)
            bytes_this_attempt += len(chunk)

            now = time.monotonic()
            if now - last_log >= 30:
                _log_progress(bytes_this_attempt, total)
                last_log = now

    _log_progress(bytes_this_attempt, total)
    print()

    final_size = os.path.getsize(part_path)
    if total is not None and final_size != total:
        raise RuntimeError(
            f"Size mismatch: got {final_size} bytes, expected {total}. "
            f"Delete {part_path} and retry."
        )

    os.rename(part_path, dest)
    print(f"  [✓] Download complete: {dest}")
    print(f"  [✓] Size: {final_size} bytes ({final_size / 1024**3:.2f} GiB)")


def _log_progress(current, total):
    if total:
        pct = current / total * 100
        msg = f"  ... {current / 1024**3:.2f} GiB / {total / 1024**3:.2f} GiB ({pct:.1f}%)"
    else:
        msg = f"  ... {current / 1024**3:.2f} GiB"
    print(msg, flush=True)


# ── SHA256 verification ───────────────────────────────────────────────

def run_sha256_verification(path, repo, filename, token):
    """Fetch LFS SHA256 from HF API, compute local file SHA256, compare.
    Returns True if match or checksum unavailable."""
    if not token:
        print("  [!] No token available, cannot fetch LFS SHA256 from HF API", file=sys.stderr)
        return False

    print("  [~] Fetching LFS SHA256 from Hugging Face API...", flush=True)
    expected = get_lfs_sha256(repo, filename, token)
    if not expected:
        print("  [!] No LFS SHA256 found in API response", file=sys.stderr)
        return False
    print(f"  [~] LFS SHA256: {expected}")

    print(f"  [~] Computing local SHA256 (this takes a while for large files)...", flush=True)
    file_size = os.path.getsize(path)
    start = time.monotonic()
    
    # Last-log tracking for progress
    last_log = [time.monotonic()]
    def progress_cb(bytes_done):
        now = time.monotonic()
        if now - last_log[0] >= 60:
            pct = bytes_done / file_size * 100 if file_size else 0
            rate = bytes_done / (now - start) / 1024**2
            eta = (file_size - bytes_done) / (rate * 1024**2) / 60 if rate > 0 else 0
            print(f"  ... {bytes_done/1024**3:.2f} GiB / {file_size/1024**3:.2f} GiB ({pct:.1f}%) @ {rate:.0f} MiB/s, ~{eta:.0f} min remaining", flush=True)
            last_log[0] = now

    digest = compute_sha256(path, progress_cb)
    elapsed = time.monotonic() - start
    print(f"  [~] Computed in {elapsed/60:.1f} min")
    print(f"  [~] Local SHA256: {digest}")

    if digest == expected:
        print(f"  [✓] SHA256 MATCHES HF LFS hash — file integrity verified")
        return True
    else:
        print(f"  [✗] SHA256 MISMATCH — file is corrupt!", file=sys.stderr)
        return False


# ── Main ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Resumable GGUF downloader with integrity verification")
    parser.add_argument("repo", help="Hugging Face repo (e.g. antirez/deepseek-v4-gguf)")
    parser.add_argument("filename", help="File name in repo")
    parser.add_argument("--out-dir", default=".", help="Output directory (default: .)")
    parser.add_argument("--token", help="HF token (default: ~/.cache/huggingface/token)")
    parser.add_argument("--force", action="store_true",
                        help="Delete any existing part/final file and restart")
    parser.add_argument("--max-retries", type=int, default=MAX_RETRIES_DEFAULT,
                        help=f"Max retries on SSL/connection errors (default: {MAX_RETRIES_DEFAULT})")
    parser.add_argument("--verify", action="store_true",
                        help="Run GGUF structural verification after download (or on existing file)")
    parser.add_argument("--verify-only", action="store_true",
                        help="Verify an existing file without downloading. Requires --out-dir.")
    parser.add_argument("--sha256", action="store_true",
                        help="Compute and verify SHA256 against HF LFS hash (takes ~1min/GiB for large files)")
    parser.add_argument("--sha256-only", action="store_true",
                        help="Compute SHA256 of an existing file and compare with HF LFS hash.")
    args = parser.parse_args()

    token = args.token
    if not token:
        token_path = os.path.expanduser("~/.cache/huggingface/token")
        if os.path.isfile(token_path):
            with open(token_path) as f:
                token = f.read().strip()

    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    dest = os.path.join(out_dir, args.filename)
    part_path = dest + ".part"

    # ── sha256-only mode ────────────────────────────────────────
    if args.sha256_only:
        path_to_check = dest if os.path.exists(dest) else part_path
        if not os.path.exists(path_to_check):
            print(f"  [!] File not found: {dest}", file=sys.stderr)
            sys.exit(1)
        ok = run_sha256_verification(path_to_check, args.repo, args.filename, token)
        sys.exit(0 if ok else 1)

    # ── verify-only mode ──────────────────────────────────────────
    if args.verify_only:
        if not os.path.exists(dest) and os.path.exists(part_path):
            print(f"  [~] Final file not found, trying .part file...")
            dest = part_path
        path_to_check = dest if os.path.exists(dest) else None
        if not path_to_check:
            print(f"  [!] File not found: {dest}", file=sys.stderr)
            sys.exit(1)
        ok, summary = verify_gguf(path_to_check)
        print(summary)
        print()
        # Also try ds4 --inspect
        ds4_out = try_ds4_inspect(path_to_check)
        if ds4_out:
            print("--- ds4 --inspect ---")
            print(ds4_out)
        sys.exit(0 if ok else 1)

    # ── normal download path ──────────────────────────────────────
    if args.force:
        for p in [dest, part_path]:
            if os.path.exists(p):
                os.remove(p)
                print(f"  [*] Removed {p}")
        print(f"  [*] Starting clean download")

    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        print(f"  [✓] Already downloaded: {dest}")
        if args.verify:
            ok, summary = verify_gguf(dest)
            print()
            print(summary)
            print()
            ds4_out = try_ds4_inspect(dest)
            if ds4_out:
                print("--- ds4 --inspect ---")
                print(ds4_out)
            if not ok:
                sys.exit(1)
        return

    url = build_url(args.repo, args.filename)
    headers = get_headers(token)

    print(f"  [~] Downloading {args.filename}")
    print(f"  [~] from {url}")
    print(f"  [~] to   {out_dir}/")

    expected_len = expected_size(url, headers)
    if expected_len:
        print(f"  [~] Expected size: {expected_len} bytes ({expected_len / 1024**3:.2f} GiB)")
    else:
        print(f"  [~] Expected size: unknown (HEAD failed)")

    if os.path.exists(dest):
        actual = os.path.getsize(dest)
        if expected_len is None or actual == expected_len:
            print(f"  [✓] Already downloaded: {dest}")
            if args.verify:
                ok, summary = verify_gguf(dest)
                print()
                print(summary)
                if not ok:
                    sys.exit(1)
            if args.sha256:
                print()
                ok = run_sha256_verification(dest, args.repo, args.filename, token)
                if not ok:
                    sys.exit(1)
            return
        else:
            print(f"  [!] Existing file size mismatch ({actual} vs {expected_len}), re-downloading")
            os.remove(dest)

    success = run_download(url, dest, part_path, headers, expected_len, args.max_retries)
    if not success:
        print(f"\n  [✗] Download failed after {args.max_retries} retries.", file=sys.stderr)
        print(f"  [~] Partial data is at: {part_path}", file=sys.stderr)
        print(f"  [~] Re-run the same command to resume.", file=sys.stderr)
        sys.exit(1)

    # Post-download verification
    if args.verify:
        print()
        print("=== GGUF structural verification ===")
        ok, summary = verify_gguf(dest)
        print(summary)
        if not ok:
            print("\n  [✗] File corrupt! Delete and re-download.", file=sys.stderr)
            sys.exit(1)

        # Try ds4 --inspect as second opinion
        ds4_out = try_ds4_inspect(dest)
        if ds4_out:
            print()
            print("--- ds4 --inspect ---")
            print(ds4_out)

    # Post-download SHA256 verification
    if args.sha256:
        print()
        print("=== SHA256 verification (this takes ~1min/GiB) ===")
        ok = run_sha256_verification(dest, args.repo, args.filename, token)
        if not ok:
            print("\n  [✗] SHA256 mismatch! File is corrupt.", file=sys.stderr)
            sys.exit(1)


if __name__ == "__main__":
    main()
