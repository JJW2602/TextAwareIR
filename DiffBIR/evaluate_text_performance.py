#!/usr/bin/env python3
"""
Evaluate text performance of restored images against SA-Text-test annotations.

This script can either consume prediction annotations already produced by the
SA-Text-style pipeline, or run the prediction portion of that pipeline from a
restored-image directory first. Supported prediction JSON examples:
- `agreed_annotations.json` from the dataset curation pipeline, where text is
  stored in `annotations[*].VLM`
- raw Bridge Spotter JSON, where text is stored in `annotations[*].rec`
- final formatted JSON, where text is stored in `entries[*].text_instances[*].text`

Keeping prediction generation and scoring as separate modes makes the evaluator
reusable for LQ, HQ, pretrained DiffBIR, and RL-finetuned outputs.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
import sys
import unicodedata
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable


TEXT_FIELD_PRIORITY = ("text", "VLM", "rec")


@dataclass(frozen=True)
class Instance:
    image_id: str
    text: str
    bbox: tuple[float, float, float, float]
    source_index: int
    score: float | None = None


@dataclass(frozen=True)
class Match:
    image_id: str
    gt_index: int
    pred_index: int
    iou: float
    gt_text: str
    pred_text: str
    normalized_gt_text: str
    normalized_pred_text: str
    exact_match: bool
    ned: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare SA-Text-style prediction annotations against SA-Text-test GT."
        )
    )
    parser.add_argument(
        "--gt-parquet",
        type=Path,
        default=Path(
            "/scratch2/james2602/TextAwareIR/SA-Text-test/data/"
            "test-00000-of-00001.parquet"
        ),
        help="Path to the SA-Text-test parquet file.",
    )
    parser.add_argument(
        "--pred-json",
        type=Path,
        default=None,
        help="Prediction JSON produced by the SA-Text-style annotation pipeline.",
    )
    parser.add_argument(
        "--restored-dir",
        type=Path,
        default=None,
        help=(
            "Directory of restored images. When provided instead of --pred-json, "
            "the script runs the SA-Text-style prediction sub-pipeline first."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where summary and detail files will be written.",
    )
    parser.add_argument(
        "--pred-text-field",
        choices=["auto", *TEXT_FIELD_PRIORITY],
        default="auto",
        help=(
            "Prediction text field to read. `auto` tries text -> VLM -> rec in that order."
        ),
    )
    parser.add_argument(
        "--pred-bbox-format",
        choices=["xyxy", "xywh"],
        default="xyxy",
        help="Bounding-box format used in the prediction JSON.",
    )
    parser.add_argument(
        "--image-id-list",
        type=Path,
        default=None,
        help=(
            "Optional newline-delimited image IDs or filenames to evaluate. "
            "Useful for per-chunk evaluation."
        ),
    )
    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=0.5,
        help="Minimum bbox IoU needed to pair a prediction with a GT instance.",
    )
    parser.add_argument(
        "--ignore-case",
        action="store_true",
        help="Lowercase texts before exact-match and NED computation.",
    )
    parser.add_argument(
        "--sa-text-pipeline-root",
        type=Path,
        default=Path("/scratch2/james2602/TextAwareIR/SA-Text_Dataset"),
        help="Root of the local SA-Text dataset-curation repository.",
    )
    parser.add_argument(
        "--sa-text-pipeline-config",
        type=Path,
        default=None,
        help=(
            "Working dataset_curation config.yaml. Required when --restored-dir "
            "is used."
        ),
    )
    parser.add_argument(
        "--pipeline-work-dir",
        type=Path,
        default=None,
        help=(
            "Directory for intermediate SA-Text-style prediction files. "
            "Defaults to <output-dir>/sa_text_pipeline."
        ),
    )
    return parser.parse_args()


def require_pyarrow() -> Any:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise SystemExit(
            "Reading SA-Text-test parquet requires `pyarrow`. "
            "Install it in the evaluation environment first, for example:\n"
            "  pip install pyarrow"
        ) from exc
    return pq


def normalize_image_id(value: str) -> str:
    return Path(value).stem


def load_image_id_filter(path: Path | None) -> set[str] | None:
    if path is None:
        return None
    image_ids: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            value = line.strip()
            if not value or value.startswith("#"):
                continue
            image_ids.add(normalize_image_id(value.split(",")[0]))
    return image_ids


def normalize_text(text: str, ignore_case: bool) -> str:
    normalized = unicodedata.normalize("NFKC", text or "")
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized.casefold() if ignore_case else normalized


def bbox_from_gt(raw_bbox: Any) -> tuple[float, float, float, float]:
    """
    Convert SA-Text bbox shape [[x1, y1], [x2, y2]] into xyxy.
    """
    if len(raw_bbox) != 2 or len(raw_bbox[0]) != 2 or len(raw_bbox[1]) != 2:
        raise ValueError(f"Unsupported GT bbox shape: {raw_bbox!r}")
    x1, y1 = raw_bbox[0]
    x2, y2 = raw_bbox[1]
    return float(x1), float(y1), float(x2), float(y2)


def bbox_from_prediction(
    raw_bbox: Any, bbox_format: str
) -> tuple[float, float, float, float]:
    if raw_bbox is None or len(raw_bbox) != 4:
        raise ValueError(f"Unsupported prediction bbox: {raw_bbox!r}")
    x1, y1, x2, y2 = map(float, raw_bbox)
    if bbox_format == "xywh":
        x2 = x1 + x2
        y2 = y1 + y2
    return x1, y1, x2, y2


def bbox_from_polygon(raw_polygon: Any) -> tuple[float, float, float, float]:
    if not raw_polygon:
        raise ValueError("Cannot build bbox from an empty polygon.")
    points = raw_polygon
    if points and not isinstance(points[0], (list, tuple)):
        if len(points) % 2 != 0:
            raise ValueError(f"Unsupported flat polygon: {raw_polygon!r}")
        points = list(zip(points[0::2], points[1::2]))
    xs = [float(point[0]) for point in points]
    ys = [float(point[1]) for point in points]
    return min(xs), min(ys), max(xs), max(ys)


def load_gt_instances(gt_parquet: Path) -> dict[str, list[Instance]]:
    pq = require_pyarrow()
    table = pq.read_table(gt_parquet, columns=["id", "text", "bbox"])
    records = table.to_pylist()

    instances_by_image: dict[str, list[Instance]] = {}
    for row in records:
        image_id = normalize_image_id(str(row["id"]))
        texts = row["text"] or []
        bboxes = row["bbox"] or []
        if len(texts) != len(bboxes):
            raise ValueError(
                f"GT text/bbox count mismatch for {image_id}: "
                f"{len(texts)} texts vs {len(bboxes)} bboxes"
            )
        image_instances = [
            Instance(
                image_id=image_id,
                text=str(text),
                bbox=bbox_from_gt(bbox),
                source_index=index,
            )
            for index, (text, bbox) in enumerate(zip(texts, bboxes))
        ]
        instances_by_image[image_id] = image_instances
    return instances_by_image


def select_prediction_text(annotation: dict[str, Any], text_field: str) -> str:
    if text_field != "auto":
        return str(annotation.get(text_field, "") or "")
    for field in TEXT_FIELD_PRIORITY:
        if field in annotation and annotation[field] is not None:
            return str(annotation[field])
    return ""


def load_prediction_instances(
    pred_json: Path, text_field: str, bbox_format: str
) -> dict[str, list[Instance]]:
    with pred_json.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    instances_by_image: dict[str, list[Instance]] = {}
    if "annotations" in payload:
        for index, annotation in enumerate(payload["annotations"]):
            file_name = annotation.get("file_name")
            if not file_name:
                continue
            bbox = annotation.get("bbox")
            if bbox is None and annotation.get("polygon") is not None:
                bbox = bbox_from_polygon(annotation["polygon"])
            else:
                bbox = bbox_from_prediction(bbox, bbox_format)
            image_id = normalize_image_id(file_name)
            instances_by_image.setdefault(image_id, []).append(
                Instance(
                    image_id=image_id,
                    text=select_prediction_text(annotation, text_field),
                    bbox=bbox,
                    source_index=index,
                    score=(
                        float(annotation["score"])
                        if annotation.get("score") is not None
                        else None
                    ),
                )
            )
        return instances_by_image

    if "entries" in payload:
        running_index = 0
        for entry in payload["entries"]:
            file_name = entry.get("original_image") or entry.get("crop_id")
            if not file_name:
                continue
            image_id = normalize_image_id(file_name)
            for text_instance in entry.get("text_instances", []):
                bbox = text_instance.get("bbox")
                if bbox is None and text_instance.get("polygon") is not None:
                    bbox = bbox_from_polygon(text_instance["polygon"])
                else:
                    bbox = bbox_from_prediction(bbox, bbox_format)
                instances_by_image.setdefault(image_id, []).append(
                    Instance(
                        image_id=image_id,
                        text=select_prediction_text(text_instance, text_field),
                        bbox=bbox,
                        source_index=running_index,
                        score=(
                            float(text_instance["score"])
                            if text_instance.get("score") is not None
                            else None
                        ),
                    )
                )
                running_index += 1
        return instances_by_image

    raise ValueError(
        "Unsupported prediction JSON schema. Expected top-level `annotations` "
        "or top-level `entries`."
    )


def import_sa_text_pipeline_modules(pipeline_root: Path) -> tuple[Any, Any, Any]:
    curation_root = pipeline_root / "dataset_curation"
    if not curation_root.is_dir():
        raise FileNotFoundError(f"dataset_curation directory not found: {curation_root}")
    sys.path.insert(0, str(curation_root))
    try:
        from src import bridge_runner, filtering, vlm_processing
    except ImportError as exc:
        raise RuntimeError(
            "Failed to import SA-Text dataset-curation modules. "
            "Run this mode from an environment that can execute the curation pipeline."
        ) from exc
    return bridge_runner, filtering, vlm_processing


def load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError(
            "Running the SA-Text-style pipeline requires PyYAML in the current environment."
        ) from exc
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def require_existing_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")


def run_sa_text_style_prediction_pipeline(
    restored_dir: Path,
    pipeline_root: Path,
    pipeline_config: Path,
    work_dir: Path,
) -> Path:
    """
    Run the text-prediction portion of the SA-Text curation pipeline.

    This mirrors the annotation branch needed for evaluation:
    Bridge -> duplicate filtering -> OVIS/Qwen recognition -> VLM filtering
    -> VLM agreement extraction.
    """
    if not restored_dir.is_dir():
        raise FileNotFoundError(f"Restored-image directory not found: {restored_dir}")
    require_existing_file(pipeline_config, "SA-Text pipeline config")
    bridge_runner, filtering, vlm_processing = import_sa_text_pipeline_modules(
        pipeline_root
    )
    config = load_yaml(pipeline_config)

    bridge_repo_dir = Path(config["bridge_repo_dir"])
    bridge_config = bridge_repo_dir / config["bridge_config_file"]
    bridge_weights = bridge_repo_dir / config["bridge_weights_file"]
    require_existing_file(bridge_config, "Bridge config")
    require_existing_file(bridge_weights, "Bridge weights")

    work_dir.mkdir(parents=True, exist_ok=True)
    bridge_dir = work_dir / "bridge"
    raw_bridge_json = bridge_runner.run_bridge(
        config=config,
        input_dir=str(restored_dir),
        output_dir=str(bridge_dir),
        stage1=False,
    )
    if raw_bridge_json is None:
        raise RuntimeError("Bridge inference failed.")

    filtered_bridge_json = work_dir / "bridge_filtered.json"
    if filtering.filter_duplicate_detections(
        raw_bridge_json,
        str(filtered_bridge_json),
        config,
    ) is None:
        raise RuntimeError("Duplicate filtering failed.")

    vlm1_name = config["vlm1_name"]
    vlm2_name = config["vlm2_name"]
    vlm1_raw_json = work_dir / f"{vlm1_name}_raw.json"
    vlm2_raw_json = work_dir / f"{vlm2_name}_raw.json"
    if vlm_processing.run_vlm_recognition(
        vlm1_name,
        str(filtered_bridge_json),
        str(restored_dir),
        str(vlm1_raw_json),
        config,
    ) is None:
        raise RuntimeError(f"{vlm1_name} recognition failed.")
    if vlm_processing.run_vlm_recognition(
        vlm2_name,
        str(filtered_bridge_json),
        str(restored_dir),
        str(vlm2_raw_json),
        config,
    ) is None:
        raise RuntimeError(f"{vlm2_name} recognition failed.")

    vlm1_filtered_json = work_dir / f"{vlm1_name}_filtered.json"
    vlm2_filtered_json = work_dir / f"{vlm2_name}_filtered.json"
    if filtering.filter_empty_vlm(
        str(vlm1_raw_json), str(vlm1_filtered_json), config
    ) is None:
        raise RuntimeError(f"{vlm1_name} filtering failed.")
    if filtering.filter_empty_vlm(
        str(vlm2_raw_json), str(vlm2_filtered_json), config
    ) is None:
        raise RuntimeError(f"{vlm2_name} filtering failed.")

    combined_json = work_dir / "vlm_combined.json"
    agreed_list = work_dir / "agreed_image_list.txt"
    agreed_json = work_dir / "agreed_annotations.json"
    if filtering.compare_and_merge_vlms(
        str(vlm1_filtered_json),
        str(vlm2_filtered_json),
        str(combined_json),
        config,
    ) is None:
        raise RuntimeError("VLM merge failed.")
    if filtering.identify_agreed_images(
        str(combined_json), str(agreed_list), config
    ) is None:
        raise RuntimeError("Agreement discovery failed.")
    if filtering.extract_agreed_annotations(
        str(combined_json), str(agreed_list), str(agreed_json), config
    ) is None:
        raise RuntimeError("Agreement extraction failed.")
    return agreed_json


def bbox_iou(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
) -> float:
    lx1, ly1, lx2, ly2 = left
    rx1, ry1, rx2, ry2 = right
    inter_x1 = max(lx1, rx1)
    inter_y1 = max(ly1, ry1)
    inter_x2 = min(lx2, rx2)
    inter_y2 = min(ly2, ry2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    intersection = inter_w * inter_h
    if intersection == 0:
        return 0.0
    left_area = max(0.0, lx2 - lx1) * max(0.0, ly2 - ly1)
    right_area = max(0.0, rx2 - rx1) * max(0.0, ry2 - ry1)
    union = left_area + right_area - intersection
    return intersection / union if union > 0 else 0.0


def levenshtein_distance(left: str, right: str) -> int:
    if left == right:
        return 0
    if not left:
        return len(right)
    if not right:
        return len(left)
    previous = list(range(len(right) + 1))
    for i, left_char in enumerate(left, start=1):
        current = [i]
        for j, right_char in enumerate(right, start=1):
            insertion = current[j - 1] + 1
            deletion = previous[j] + 1
            substitution = previous[j - 1] + (left_char != right_char)
            current.append(min(insertion, deletion, substitution))
        previous = current
    return previous[-1]


def normalized_edit_similarity(left: str, right: str) -> float:
    denominator = max(len(left), len(right), 1)
    return 1.0 - levenshtein_distance(left, right) / denominator


def match_instances(
    image_id: str,
    gt_instances: list[Instance],
    pred_instances: list[Instance],
    iou_threshold: float,
    ignore_case: bool,
) -> tuple[list[Match], list[int], list[int]]:
    candidate_pairs: list[tuple[float, int, int]] = []
    for gt_index, gt_instance in enumerate(gt_instances):
        for pred_index, pred_instance in enumerate(pred_instances):
            iou = bbox_iou(gt_instance.bbox, pred_instance.bbox)
            if iou >= iou_threshold:
                candidate_pairs.append((iou, gt_index, pred_index))
    candidate_pairs.sort(reverse=True)

    matched_gt: set[int] = set()
    matched_pred: set[int] = set()
    matches: list[Match] = []
    for iou, gt_index, pred_index in candidate_pairs:
        if gt_index in matched_gt or pred_index in matched_pred:
            continue
        gt_instance = gt_instances[gt_index]
        pred_instance = pred_instances[pred_index]
        normalized_gt = normalize_text(gt_instance.text, ignore_case)
        normalized_pred = normalize_text(pred_instance.text, ignore_case)
        matches.append(
            Match(
                image_id=image_id,
                gt_index=gt_index,
                pred_index=pred_index,
                iou=iou,
                gt_text=gt_instance.text,
                pred_text=pred_instance.text,
                normalized_gt_text=normalized_gt,
                normalized_pred_text=normalized_pred,
                exact_match=normalized_gt == normalized_pred,
                ned=normalized_edit_similarity(normalized_gt, normalized_pred),
            )
        )
        matched_gt.add(gt_index)
        matched_pred.add(pred_index)

    unmatched_gt = [
        index for index in range(len(gt_instances)) if index not in matched_gt
    ]
    unmatched_pred = [
        index for index in range(len(pred_instances)) if index not in matched_pred
    ]
    return matches, unmatched_gt, unmatched_pred


def safe_divide(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def f1_score(precision: float, recall: float) -> float:
    return (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )


def write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    rows = list(rows)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def evaluate(
    gt_by_image: dict[str, list[Instance]],
    pred_by_image: dict[str, list[Instance]],
    iou_threshold: float,
    ignore_case: bool,
) -> tuple[dict[str, Any], list[Match], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    all_image_ids = sorted(set(gt_by_image) | set(pred_by_image))
    all_matches: list[Match] = []
    unmatched_gt_rows: list[dict[str, Any]] = []
    unmatched_pred_rows: list[dict[str, Any]] = []
    per_image_rows: list[dict[str, Any]] = []

    total_gt = 0
    total_pred = 0
    total_detection_matches = 0
    total_exact_matches = 0

    for image_id in all_image_ids:
        gt_instances = gt_by_image.get(image_id, [])
        pred_instances = pred_by_image.get(image_id, [])
        matches, unmatched_gt, unmatched_pred = match_instances(
            image_id=image_id,
            gt_instances=gt_instances,
            pred_instances=pred_instances,
            iou_threshold=iou_threshold,
            ignore_case=ignore_case,
        )
        all_matches.extend(matches)

        total_gt += len(gt_instances)
        total_pred += len(pred_instances)
        total_detection_matches += len(matches)
        total_exact_matches += sum(match.exact_match for match in matches)

        per_image_rows.append(
            {
                "image_id": image_id,
                "gt_instances": len(gt_instances),
                "pred_instances": len(pred_instances),
                "matched_instances": len(matches),
                "exact_matches": sum(match.exact_match for match in matches),
                "mean_ned_matched": (
                    statistics.mean(match.ned for match in matches) if matches else 0.0
                ),
            }
        )

        for index in unmatched_gt:
            unmatched_gt_rows.append(asdict(gt_instances[index]))
        for index in unmatched_pred:
            unmatched_pred_rows.append(asdict(pred_instances[index]))

    det_precision = safe_divide(total_detection_matches, total_pred)
    det_recall = safe_divide(total_detection_matches, total_gt)
    e2e_precision = safe_divide(total_exact_matches, total_pred)
    e2e_recall = safe_divide(total_exact_matches, total_gt)
    recognition_exact_on_matched = safe_divide(
        total_exact_matches, total_detection_matches
    )
    mean_ned_on_matched = (
        statistics.mean(match.ned for match in all_matches) if all_matches else 0.0
    )

    summary = {
        "num_images_gt": len(gt_by_image),
        "num_images_pred": len(pred_by_image),
        "num_gt_instances": total_gt,
        "num_pred_instances": total_pred,
        "num_detection_matches": total_detection_matches,
        "num_exact_text_matches": total_exact_matches,
        "iou_threshold": iou_threshold,
        "detection": {
            "precision": det_precision,
            "recall": det_recall,
            "f1": f1_score(det_precision, det_recall),
        },
        "end_to_end_exact": {
            "precision": e2e_precision,
            "recall": e2e_recall,
            "f1": f1_score(e2e_precision, e2e_recall),
        },
        "recognition_on_matched_detections": {
            "exact_match_accuracy": recognition_exact_on_matched,
            "mean_ned": mean_ned_on_matched,
        },
        "unmatched": {
            "gt_instances": len(unmatched_gt_rows),
            "pred_instances": len(unmatched_pred_rows),
        },
    }
    return (
        summary,
        all_matches,
        unmatched_gt_rows,
        unmatched_pred_rows,
        per_image_rows,
    )


def main() -> None:
    args = parse_args()
    if not 0 <= args.iou_threshold <= 1:
        raise SystemExit("--iou-threshold must be between 0 and 1.")
    if (args.pred_json is None) == (args.restored_dir is None):
        raise SystemExit("Provide exactly one of --pred-json or --restored-dir.")
    if args.restored_dir is not None and args.sa_text_pipeline_config is None:
        raise SystemExit(
            "--sa-text-pipeline-config is required when --restored-dir is used."
        )

    pred_json = args.pred_json
    if args.restored_dir is not None:
        pipeline_work_dir = args.pipeline_work_dir or (
            args.output_dir / "sa_text_pipeline"
        )
        pred_json = run_sa_text_style_prediction_pipeline(
            restored_dir=args.restored_dir,
            pipeline_root=args.sa_text_pipeline_root,
            pipeline_config=args.sa_text_pipeline_config,
            work_dir=pipeline_work_dir,
        )

    gt_by_image = load_gt_instances(args.gt_parquet)
    pred_by_image = load_prediction_instances(
        pred_json=pred_json,
        text_field=args.pred_text_field,
        bbox_format=args.pred_bbox_format,
    )
    image_id_filter = load_image_id_filter(args.image_id_list)
    if image_id_filter is not None:
        gt_by_image = {
            image_id: gt_by_image.get(image_id, [])
            for image_id in sorted(image_id_filter)
        }
        pred_by_image = {
            image_id: pred_by_image.get(image_id, [])
            for image_id in sorted(image_id_filter)
        }
    (
        summary,
        matches,
        unmatched_gt_rows,
        unmatched_pred_rows,
        per_image_rows,
    ) = evaluate(
        gt_by_image=gt_by_image,
        pred_by_image=pred_by_image,
        iou_threshold=args.iou_threshold,
        ignore_case=args.ignore_case,
    )
    if image_id_filter is not None:
        summary["image_id_filter_count"] = len(image_id_filter)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    write_csv(
        args.output_dir / "matched_instances.csv",
        [asdict(match) for match in matches],
    )
    write_csv(args.output_dir / "unmatched_gt.csv", unmatched_gt_rows)
    write_csv(args.output_dir / "unmatched_pred.csv", unmatched_pred_rows)
    write_csv(args.output_dir / "per_image.csv", per_image_rows)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
