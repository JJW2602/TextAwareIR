#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pyarrow.parquet as pq


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--count", type=int, default=10_000)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--prefix", type=str, default="sa_text_train_10000")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = args.source_dir / "data"
    shards = sorted(data_dir.glob("train-*.parquet"))
    if not shards:
        raise FileNotFoundError(f"No train parquet shards found under {data_dir}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out_dir / f"{args.prefix}.jsonl"
    ids_path = args.out_dir / f"{args.prefix}_ids.txt"
    summary_path = args.out_dir / f"{args.prefix}_summary.json"

    selected: list[dict[str, object]] = []
    for shard in shards:
        if len(selected) >= args.count:
            break
        table = pq.read_table(shard, columns=["id"])
        ids = table.column("id").to_pylist()
        for row_index, image_id in enumerate(ids):
            if len(selected) >= args.count:
                break
            selected.append(
                {
                    "sample_index": len(selected),
                    "id": str(image_id),
                    "parquet": str(shard),
                    "row_index": row_index,
                }
            )

    if len(selected) != args.count:
        raise RuntimeError(f"Requested {args.count} samples, but only found {len(selected)}")

    with manifest_path.open("w", encoding="utf-8") as handle:
        for entry in selected:
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")

    with ids_path.open("w", encoding="utf-8") as handle:
        for entry in selected:
            handle.write(str(entry["id"]) + "\n")

    used_counts: dict[str, int] = {}
    for entry in selected:
        shard = str(entry["parquet"])
        used_counts[shard] = used_counts.get(shard, 0) + 1

    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_dir": str(args.source_dir),
        "count": len(selected),
        "manifest": str(manifest_path),
        "ids": str(ids_path),
        "shards": used_counts,
        "first_id": selected[0]["id"],
        "last_id": selected[-1]["id"],
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
