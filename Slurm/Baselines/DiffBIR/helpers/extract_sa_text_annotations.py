#!/usr/bin/env python3
"""Extract SA-Text-style text annotations from DiffBIR restored images."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import shutil
import sys
from pathlib import Path
from typing import Any


IMAGE_EXTS = {".png", ".jpg", ".jpeg"}


def repo_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "DiffBIR").is_dir() and (parent / "SA-Text_Dataset").is_dir():
            return parent
    raise RuntimeError("Could not locate TextAwareIR repository root.")


def parse_args() -> argparse.Namespace:
    root = repo_root()
    default_results = root / "DiffBIR/results/sa_text_test/lv2_2gpu"
    default_output = default_results / "text_annotations"
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, default=default_results)
    parser.add_argument(
        "--chunks",
        nargs="+",
        default=["chunk_0", "chunk_1"],
        help="Chunk directories under --results-root.",
    )
    parser.add_argument("--output-dir", type=Path, default=default_output)
    parser.add_argument("--flat-input-dir", type=Path, default=None)
    parser.add_argument(
        "--config",
        type=Path,
        default=root / "DiffBIR/slurm/helpers/sa_text_annotation_config.yaml",
    )
    parser.add_argument(
        "--pipeline-root",
        type=Path,
        default=root / "SA-Text_Dataset",
    )
    parser.add_argument("--work-dir", type=Path, default=None)
    parser.add_argument(
        "--reuse-existing",
        action="store_true",
        help="Skip model inference when the agreed annotation JSON already exists.",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Only create the flat input symlinks and manifest; do not run models.",
    )
    return parser.parse_args()


def collect_images(results_root: Path, chunks: list[str]) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    seen: dict[str, Path] = {}
    for chunk in chunks:
        chunk_dir = results_root / chunk
        if not chunk_dir.is_dir():
            raise FileNotFoundError(f"Chunk directory not found: {chunk_dir}")
        for path in sorted(chunk_dir.iterdir()):
            if not path.is_file() or path.suffix.lower() not in IMAGE_EXTS:
                continue
            if path.name in seen and seen[path.name] != path:
                raise ValueError(
                    f"Duplicate image filename across chunks: {path.name}\n"
                    f"  first: {seen[path.name]}\n"
                    f"  second: {path}"
                )
            seen[path.name] = path
            records.append(
                {
                    "file_name": path.name,
                    "chunk": chunk,
                    "source_path": str(path),
                }
            )
    if not records:
        raise RuntimeError(f"No images found under {results_root} chunks {chunks}")
    return records


def prepare_flat_input(records: list[dict[str, str]], flat_input_dir: Path) -> None:
    flat_input_dir.mkdir(parents=True, exist_ok=True)
    for record in records:
        source = Path(record["source_path"])
        target = flat_input_dir / record["file_name"]
        if target.exists() or target.is_symlink():
            if target.is_symlink() and target.resolve() == source.resolve():
                continue
            raise FileExistsError(
                f"Flat input target already exists and is not the expected symlink: {target}"
            )
        os.symlink(source, target)


def write_manifest(records: list[dict[str, str]], manifest_path: Path) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["file_name", "chunk", "source_path"])
        writer.writeheader()
        writer.writerows(records)


def write_image_id_list(records: list[dict[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(f"{Path(record['file_name']).stem}\n")


def annotation_text(annotation: dict[str, Any]) -> str:
    for key in ("VLM", "text", "OVIS", "Qwen", "rec"):
        value = annotation.get(key)
        if value is not None:
            return str(value)
    return ""


def write_annotation_csv(json_path: Path, csv_path: Path) -> None:
    if not json_path.is_file():
        return
    with json_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    rows = []
    for ann in payload.get("annotations", []):
        rows.append(
            {
                "file_name": ann.get("file_name", ""),
                "annotation_id": ann.get("id", ""),
                "text": annotation_text(ann),
                "OVIS": ann.get("OVIS", ""),
                "Qwen": ann.get("Qwen", ""),
                "VLM": ann.get("VLM", ""),
                "has_text": ann.get("has_text", ""),
                "score": ann.get("score", ""),
                "bbox": json.dumps(ann.get("bbox", []), ensure_ascii=False),
                "polygon": json.dumps(ann.get("polygon", []), ensure_ascii=False),
            }
        )
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "file_name",
            "annotation_id",
            "text",
            "OVIS",
            "Qwen",
            "VLM",
            "has_text",
            "score",
            "bbox",
            "polygon",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def import_prediction_runner() -> Any:
    diffbir_dir = repo_root() / "DiffBIR"
    sys.path.insert(0, str(diffbir_dir))
    from evaluate_text_performance import run_sa_text_style_prediction_pipeline

    return run_sa_text_style_prediction_pipeline


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    output_dir = args.output_dir
    flat_input_dir = args.flat_input_dir or (output_dir / "flat_input")
    work_dir = args.work_dir or (output_dir / "pipeline")
    agreed_json = work_dir / "agreed_annotations.json"

    records = collect_images(args.results_root, args.chunks)
    prepare_flat_input(records, flat_input_dir)
    write_manifest(records, output_dir / "input_manifest.csv")
    write_image_id_list(records, output_dir / "image_id_list.txt")
    logging.info("Prepared %d images in %s", len(records), flat_input_dir)

    if args.prepare_only:
        logging.info("Prepare-only mode complete.")
        return

    if args.reuse_existing and agreed_json.is_file():
        logging.info("Reusing existing annotations: %s", agreed_json)
        pred_json = agreed_json
    else:
        run_prediction_pipeline = import_prediction_runner()
        pred_json = Path(
            run_prediction_pipeline(
                restored_dir=flat_input_dir,
                pipeline_root=args.pipeline_root,
                pipeline_config=args.config,
                work_dir=work_dir,
            )
        )

    final_json = output_dir / "agreed_annotations.json"
    if pred_json.resolve() != final_json.resolve():
        shutil.copy2(pred_json, final_json)

    write_annotation_csv(final_json, output_dir / "agreed_annotations.csv")
    write_annotation_csv(work_dir / "vlm_combined.json", output_dir / "vlm_combined.csv")

    logging.info("Done.")
    logging.info("Agreed annotations JSON: %s", final_json)
    logging.info("Agreed annotations CSV: %s", output_dir / "agreed_annotations.csv")
    logging.info("Combined VLM CSV: %s", output_dir / "vlm_combined.csv")


if __name__ == "__main__":
    main()
