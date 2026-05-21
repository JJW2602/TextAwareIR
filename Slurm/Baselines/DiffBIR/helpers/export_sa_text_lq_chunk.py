#!/usr/bin/env python3
"""Export one SA-Text-test LQ chunk from parquet to an image directory."""

from __future__ import annotations

import argparse
import csv
import io
import re
from pathlib import Path
from typing import Any

from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet", type=Path, required=True)
    parser.add_argument("--level", type=int, choices=(1, 2, 3), default=2)
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def import_pyarrow() -> Any:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise SystemExit(
            "pyarrow is required to read SA-Text-test parquet files. Install it in "
            "the DiffBIR env, for example:\n"
            "  /home/james2602/miniconda3/envs/diffbir/bin/python -m pip install pyarrow"
        ) from exc
    return pq


def safe_stem(value: str) -> str:
    stem = Path(str(value)).stem
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._")
    return stem or "sample"


def image_from_hf_value(value: Any) -> Image.Image:
    if isinstance(value, dict):
        if value.get("bytes") is not None:
            return Image.open(io.BytesIO(value["bytes"])).convert("RGB")
        if value.get("path"):
            return Image.open(value["path"]).convert("RGB")
    if isinstance(value, (bytes, bytearray)):
        return Image.open(io.BytesIO(value)).convert("RGB")
    raise ValueError(f"Unsupported image payload type: {type(value)!r}")


def image_size_from_hf_value(value: Any) -> tuple[int | None, int | None]:
    try:
        with image_from_hf_value(value) as image:
            return image.size
    except Exception:
        return None, None


def main() -> None:
    args = parse_args()
    if args.start < 0:
        raise SystemExit("--start must be non-negative.")
    if args.count <= 0:
        raise SystemExit("--count must be positive.")

    pq = import_pyarrow()
    lq_column = f"lq_img_lv{args.level}"
    columns = ["id", lq_column, "hq_img", "text", "bbox"]
    table = pq.read_table(args.parquet, columns=columns)
    total_rows = table.num_rows
    end = min(args.start + args.count, total_rows)
    if args.start >= total_rows:
        raise SystemExit(
            f"Chunk start {args.start} is outside dataset with {total_rows} rows."
        )

    rows = table.slice(args.start, end - args.start).to_pylist()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)

    manifest_rows: list[dict[str, Any]] = []
    for global_index, row in enumerate(rows, start=args.start):
        image_id = str(row["id"])
        file_name = f"{safe_stem(image_id)}.png"
        output_path = args.output_dir / file_name

        if args.skip_existing and output_path.is_file():
            lq_w, lq_h = image_size_from_hf_value(row[lq_column])
        else:
            image = image_from_hf_value(row[lq_column])
            lq_w, lq_h = image.size
            image.save(output_path)

        hq_w, hq_h = image_size_from_hf_value(row["hq_img"])
        manifest_rows.append(
            {
                "global_index": global_index,
                "id": image_id,
                "file_name": file_name,
                "level": args.level,
                "lq_width": lq_w,
                "lq_height": lq_h,
                "hq_width": hq_w,
                "hq_height": hq_h,
                "num_text": len(row.get("text") or []),
                "num_bbox": len(row.get("bbox") or []),
            }
        )

    with args.manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest_rows[0].keys()))
        writer.writeheader()
        writer.writerows(manifest_rows)

    print(
        f"exported {len(manifest_rows)} lv{args.level} images "
        f"[{args.start}, {end}) to {args.output_dir}"
    )


if __name__ == "__main__":
    main()
