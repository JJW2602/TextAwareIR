#!/usr/bin/env python3
"""Normalize TAIR native output names for SA-Text evaluation.

TAIR infer.py writes restored images as ``restored_<image_id>.png``.  The
SA-Text GT and the shared annotation/evaluation helpers expect ``<image_id>``.
This helper creates a clean chunk directory with matching stems, usually by
symlinking the TAIR files to names without the ``restored_`` prefix.
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
from pathlib import Path


IMAGE_EXTS = {".png", ".jpg", ".jpeg"}
RESTORED_PREFIX = "restored_"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--normalized-root", type=Path, required=True)
    parser.add_argument("--chunks", nargs="+", default=["chunk_0", "chunk_1"])
    parser.add_argument(
        "--copy",
        action="store_true",
        help="Copy files instead of creating symlinks.",
    )
    return parser.parse_args()


def normalize_chunk(raw_root: Path, normalized_root: Path, chunk: str, copy: bool) -> int:
    raw_dir = raw_root / chunk
    out_dir = normalized_root / chunk
    if not raw_dir.is_dir():
        raise FileNotFoundError(f"Raw TAIR chunk directory not found: {raw_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, str]] = []
    for source in sorted(raw_dir.iterdir()):
        if not source.is_file() or source.suffix.lower() not in IMAGE_EXTS:
            continue
        if not source.stem.startswith(RESTORED_PREFIX):
            continue
        image_id = source.stem[len(RESTORED_PREFIX) :]
        target = out_dir / f"{image_id}{source.suffix.lower()}"
        if target.exists() or target.is_symlink():
            if target.is_symlink() and target.resolve() == source.resolve():
                pass
            else:
                target.unlink()
        if not target.exists():
            if copy:
                shutil.copy2(source, target)
            else:
                os.symlink(source, target)
        rows.append(
            {
                "chunk": chunk,
                "image_id": image_id,
                "raw_path": str(source),
                "normalized_path": str(target),
            }
        )

    if not rows:
        raise RuntimeError(f"No restored_* image files found in {raw_dir}")

    with (out_dir / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    return len(rows)


def main() -> None:
    args = parse_args()
    total = 0
    for chunk in args.chunks:
        count = normalize_chunk(args.raw_root, args.normalized_root, chunk, args.copy)
        total += count
        print(f"{chunk}: normalized {count} TAIR restored images")
    print(f"Output: {args.normalized_root}")
    print(f"Total normalized: {total}")


if __name__ == "__main__":
    main()
