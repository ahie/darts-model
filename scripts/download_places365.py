#!/usr/bin/env python3
"""Download Places365-Standard (256x256 train split) into a flat directory.

The renderer composites these photographs behind the board and uses them as
the environment it reflects, so the model sees real-image statistics around a
synthetic board. Both configs point ``bg_image_dir`` at ``bg_images``.

About 1.8M images: a 24 GB download, and roughly the same again once
extracted. The tar can be deleted afterwards. Both steps resume if
interrupted.

Places365 is distributed by MIT CSAIL for non-commercial research and
education; see http://places2.csail.mit.edu for its terms.

Usage:
    python scripts/download_places365.py [--output-dir bg_images]
"""
from __future__ import annotations

import argparse
import os
import sys
import tarfile
import time
import urllib.request

URL = "http://data.csail.mit.edu/places/places365/train_256_places365standard.tar"
IMAGE_EXTS = (".jpg", ".jpeg", ".png")


def download(url: str, dest: str) -> None:
    partial = dest + ".partial"
    have = os.path.getsize(partial) if os.path.exists(partial) else 0
    req = urllib.request.Request(url)
    if have:
        req.add_header("Range", f"bytes={have}-")
        print(f"Resuming {url} from {have / 1e9:.1f} GB")
    else:
        print(f"Downloading {url}")

    with urllib.request.urlopen(req) as resp:
        # A server that ignores Range answers 200 with the whole file, which
        # appended to a partial would corrupt it.
        if have and resp.status != 206:
            have = 0
            mode = "wb"
        else:
            mode = "ab"
        length = resp.headers.get("Content-Length")
        total = int(length) + have if length else None
        got = have
        start = time.time()
        with open(partial, mode) as f:
            while chunk := resp.read(1 << 20):
                f.write(chunk)
                got += len(chunk)
                rate = (got - have) / max(time.time() - start, 1e-6) / 1e6
                if total:
                    print(f"\r  {100 * got / total:.0f}% ({got / 1e9:.1f}/"
                          f"{total / 1e9:.1f} GB) {rate:.1f} MB/s",
                          end="", flush=True)
                else:
                    print(f"\r  {got / 1e9:.1f} GB {rate:.1f} MB/s",
                          end="", flush=True)
    print()
    if total and got < total:
        sys.exit(f"Incomplete download ({got}/{total} bytes). Re-run to resume.")
    os.replace(partial, dest)


def extract(tar_path: str, out_dir: str) -> None:
    """Flatten every image into out_dir as 0000000.jpg, 0000001.jpg, ...

    Sequential names because Places365 reuses file names across its category
    directories, and the renderer reads a single flat directory. Resuming skips
    as many images as are already present, which relies on the tar's order
    being stable -- it is, for an unmodified file.

    Each image is written to a temporary name and renamed into place, so an
    interrupted run never leaves a truncated JPEG under a final name. The last
    existing image is extracted again on resume regardless, which also repairs
    a directory written without that guarantee.
    """
    os.makedirs(out_dir, exist_ok=True)
    have = sum(1 for f in os.listdir(out_dir) if f.endswith(".jpg"))
    resume_at = max(have - 1, 0)
    if have:
        print(f"  {have} images already in {out_dir}/, resuming from "
              f"{resume_at:07d}.jpg")
    idx = written = 0
    start = time.time()
    with tarfile.open(tar_path, "r") as tar:
        for member in tar:
            if not (member.isfile() and member.name.lower().endswith(IMAGE_EXTS)):
                continue
            if idx < resume_at:
                idx += 1
                continue
            dest = os.path.join(out_dir, f"{idx:07d}.jpg")
            with tar.extractfile(member) as src, open(dest + ".part", "wb") as dst:
                dst.write(src.read())
            os.replace(dest + ".part", dest)
            idx += 1
            written += 1
            if written % 50000 == 0:
                print(f"  {written} extracted ({time.time() - start:.0f}s)",
                      flush=True)
    print(f"Done: {idx} images in {out_dir}/")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--output-dir", default="bg_images")
    ap.add_argument("--tar", default="train_256_places365standard.tar",
                    help="where to keep the downloaded archive")
    args = ap.parse_args()

    if os.path.exists(args.tar):
        print(f"{args.tar} exists, skipping download")
    else:
        download(URL, args.tar)
    extract(args.tar, args.output_dir)
    print(f"{args.tar} can now be deleted to free "
          f"{os.path.getsize(args.tar) / 1e9:.1f} GB")


if __name__ == "__main__":
    main()
