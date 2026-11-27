#!/usr/bin/env python3
"""Download and unpack the DiffSinger OpenCpop ONNX model (~220 MB).

    python scripts/fetch_model.py                 # download from GitHub releases
    python scripts/fetch_model.py --zip model.zip # use an already downloaded file

The zip is the official release asset of
https://github.com/HighCWu/DiffSinger-ONNX-Infer (MIT, see vendor/diffsinger_onnx).

Files are unpacked to models/opencpop/ (git-ignored).  Nothing else is
downloaded; the inference code is vendored in this repository.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import urllib.request
import zipfile

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_URL = "https://github.com/HighCWu/DiffSinger-ONNX-Infer/releases/download/v0.1/model.zip"
EXPECTED_SHA256 = "598739ffde0b10a51dfef74ab0ec1eb2820941f4b81ada64079b733e895c9c77"
EXPECTED_SIZE = 220222146


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download(url, target):
    os.makedirs(os.path.dirname(target), exist_ok=True)

    def progress(blocks, block_size, total):
        done = blocks * block_size
        if total > 0:
            sys.stdout.write("\r%5.1f%%  %.1f / %.1f MB" % (
                min(100.0, done * 100.0 / total), done / 1e6, total / 1e6))
            sys.stdout.flush()

    urllib.request.urlretrieve(url, target, progress)
    print()


def main(argv=None):
    parser = argparse.ArgumentParser(description="Fetch the OpenCpop ONNX model")
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--zip", default=None, help="use a local model.zip instead of downloading")
    parser.add_argument("--dest", default=os.path.join(REPO_ROOT, "models", "opencpop"))
    parser.add_argument("--force", action="store_true", help="re-extract even if files exist")
    parser.add_argument("--skip-hash", action="store_true", help="skip the sha256 check")
    args = parser.parse_args(argv)

    if not args.force and os.path.isfile(os.path.join(args.dest, "fs2.onnx")):
        print("model already present in %s (use --force to re-extract)" % args.dest)
        return 0

    zip_path = args.zip
    if zip_path is None:
        zip_path = os.path.join(REPO_ROOT, "models", "_download", "model.zip")
        if not os.path.isfile(zip_path) or os.path.getsize(zip_path) != EXPECTED_SIZE:
            print("downloading %s" % args.url)
            download(args.url, zip_path)

    size = os.path.getsize(zip_path)
    print("archive: %s (%.1f MB)" % (zip_path, size / 1e6))
    if not args.skip_hash:
        digest = sha256(zip_path)
        if digest.lower() != EXPECTED_SHA256:
            print("sha256 mismatch:\n  got      %s\n  expected %s" % (digest, EXPECTED_SHA256))
            return 1
        print("sha256 ok")

    os.makedirs(args.dest, exist_ok=True)
    with zipfile.ZipFile(zip_path) as archive:
        for info in archive.infolist():
            name = os.path.basename(info.filename)
            if not name or name == ".gitkeep":
                continue
            target = os.path.join(args.dest, name)
            with archive.open(info) as source, open(target, "wb") as sink:
                sink.write(source.read())
            print("  %-16s %10.1f MB" % (name, os.path.getsize(target) / 1e6))

    print("model ready in %s" % args.dest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
