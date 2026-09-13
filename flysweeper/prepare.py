"""Download the three MaleCNS v1.0 flat-connectome tables and verify their SHA-256.

Source: https://male-cns.janelia.org/download/  (CC BY 4.0, Berg et al., Cell 2026)
Bucket: gs://flyem-male-cns/v1.0/connectome-data/flat-connectome/

Usage:
    python -m flysweeper.prepare            # download missing files, verify all
    python -m flysweeper.prepare --verify   # verify only, no downloads
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import urllib.request
from pathlib import Path

from .paths import RAW, ensure_dirs

BUCKET = "https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome/"

# Byte counts and digests were checked against the live bucket on 2026-09-13.
FILES = {
    "annotations": {
        "filename": "body-annotations-male-cns-v1.0-minconf-0.5.feather",
        "bytes": 14483314,
        "sha256": "2177e246113e4cfbf1e7772ec37c6da1955ff22e8063d0b1f833101f99a9a3b2",
    },
    "neurotransmitters": {
        "filename": "body-neurotransmitters-male-cns-v1.0.feather",
        "bytes": 43282834,
        "sha256": "95c9289220663abeb3409f3ad9e5a7f8a53f8093f5139d15502cd08da8879621",
    },
    "weights": {
        "filename": "connectome-weights-male-cns-v1.0-minconf-0.5.feather",
        "bytes": 1051241946,
        "sha256": "e35da783d1c686b2b58b3b87cd6a403ae43bfcfba8bff28e08ef752c1a56afc1",
    },
}


def raw_path(key: str) -> Path:
    return RAW / FILES[key]["filename"]


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url: str, target: Path) -> None:
    partial = target.with_suffix(target.suffix + ".part")
    start = partial.stat().st_size if partial.exists() else 0
    req = urllib.request.Request(url)
    if start:
        req.add_header("Range", f"bytes={start}-")
    t0 = time.time()
    with urllib.request.urlopen(req) as resp, partial.open("ab" if start else "wb") as out:
        total = start + int(resp.headers.get("Content-Length", 0))
        done = start
        while True:
            chunk = resp.read(1 << 22)
            if not chunk:
                break
            out.write(chunk)
            done += len(chunk)
            if total:
                pct = 100.0 * done / total
                rate = (done - start) / max(time.time() - t0, 1e-6) / 1e6
                print(f"\r  {target.name}: {pct:5.1f}%  {done/1e6:8.1f} MB  {rate:6.1f} MB/s", end="", flush=True)
    print()
    partial.replace(target)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--verify", action="store_true", help="only verify existing files")
    args = ap.parse_args(argv)
    ensure_dirs()

    ok = True
    lock = {}
    for key, spec in FILES.items():
        target = raw_path(key)
        url = BUCKET + spec["filename"]
        if not target.exists():
            if args.verify:
                print(f"MISSING  {target.name}")
                ok = False
                continue
            print(f"download {url}")
            download(url, target)
        size = target.stat().st_size
        if size != spec["bytes"]:
            print(f"SIZE MISMATCH {target.name}: {size} != {spec['bytes']}")
            ok = False
            continue
        digest = sha256_of(target)
        if digest != spec["sha256"]:
            print(f"SHA256 MISMATCH {target.name}: {digest}")
            ok = False
            continue
        print(f"ok       {target.name}  ({size/1e6:.1f} MB)")
        lock[spec["filename"]] = {"url": url, "bytes": size, "sha256": digest}

    if ok:
        (RAW / "source.lock.json").write_text(json.dumps(lock, indent=2) + "\n")
        print(f"wrote {RAW / 'source.lock.json'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
