#!/usr/bin/env python3
"""Create side-by-side GT and DiffBIR visualization images for SA-Text chunks."""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--parquet",
        type=Path,
        default=Path(
            "/scratch2/james2602/TextAwareIR/Dataset/SA-Text-test/data/"
            "test-00000-of-00001.parquet"
        ),
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path(
            "/scratch2/james2602/TextAwareIR/Results/Baselines/DiffBIR/"
            "sa_text_test_lv2_a6000"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Defaults to <results-root>/GT_DiffBIR_inference.",
    )
    parser.add_argument(
        "--chunks",
        nargs="+",
        default=["chunk_0", "chunk_1"],
        help="Chunk directories to visualize, e.g. chunk_0 chunk_1.",
    )
    parser.add_argument("--max-images-per-chunk", type=int, default=None)
    parser.add_argument("--no-bboxes", action="store_true")
    parser.add_argument(
        "--bridge-json-fmt",
        type=str,
        default="{results_root}/text_annotations/{chunk}/pipeline/bridge_filtered.json",
        help="Format string for per-chunk Bridge prediction JSON. "
             "Available keys: {results_root}, {chunk}. "
             "Set to empty string to skip Bridge overlay.",
    )
    parser.add_argument(
        "--bridge-score-threshold",
        type=float,
        default=0.3,
        help="Minimum Bridge confidence score to draw (default 0.3).",
    )
    parser.add_argument("--contact-sheet-cols", type=int, default=4)
    parser.add_argument("--contact-sheet-rows", type=int, default=5)
    parser.add_argument("--thumb-width", type=int, default=360)
    return parser.parse_args()


def import_pyarrow() -> Any:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise SystemExit(
            "pyarrow is required to read SA-Text-test parquet files. "
            "Use the DiffBIR env or install pyarrow."
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


def load_gt_rows(parquet_path: Path) -> dict[str, dict[str, Any]]:
    pq = import_pyarrow()
    table = pq.read_table(parquet_path, columns=["id", "hq_img", "text", "bbox"])
    rows: dict[str, dict[str, Any]] = {}
    for row in table.to_pylist():
        image_id = safe_stem(str(row["id"]))
        rows[image_id] = row
    return rows


def load_font(size: int) -> ImageFont.ImageFont:
    candidates = [
        "/usr/share/fonts/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
    ]
    for candidate in candidates:
        path = Path(candidate)
        if path.is_file():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def draw_label(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    font: ImageFont.ImageFont,
    fill: tuple[int, int, int] = (255, 255, 255),
) -> None:
    x, y = xy
    bbox = draw.textbbox((x, y), text, font=font)
    pad = 4
    draw.rectangle(
        (bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad),
        fill=(0, 0, 0),
    )
    draw.text((x, y), text, font=font, fill=fill)


def load_bridge_predictions(
    json_path: Path,
    score_threshold: float,
) -> dict[str, list[dict[str, Any]]]:
    if not json_path.is_file():
        return {}
    with json_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    annotations = payload.get("annotations") if isinstance(payload, dict) else None
    if not annotations:
        return {}
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for ann in annotations:
        file_name = ann.get("file_name")
        if not file_name:
            continue
        score = float(ann.get("score", 0.0))
        if score < score_threshold:
            continue
        grouped[safe_stem(file_name)].append(ann)
    return grouped


def draw_bridge_annotations(
    image: Image.Image,
    annotations: list[dict[str, Any]],
    font: ImageFont.ImageFont,
) -> None:
    draw = ImageDraw.Draw(image)
    width, height = image.size
    for ann in annotations:
        bbox = ann.get("bbox")
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            continue
        x1, y1, x2, y2 = (float(v) for v in bbox)
        x1 = max(0, min(width - 1, int(round(x1))))
        y1 = max(0, min(height - 1, int(round(y1))))
        x2 = max(0, min(width - 1, int(round(x2))))
        y2 = max(0, min(height - 1, int(round(y2))))
        if x2 <= x1 or y2 <= y1:
            continue
        draw.rectangle((x1, y1, x2, y2), outline=(40, 160, 255), width=2)
        rec = str(ann.get("rec") or "").strip()
        score = ann.get("score")
        if rec or score is not None:
            label_parts = []
            if rec:
                label_parts.append(rec if len(rec) <= 24 else rec[:21] + "...")
            if isinstance(score, (int, float)):
                label_parts.append(f"{float(score):.2f}")
            label = " ".join(label_parts)
            label_y = y2 + 2 if y2 + 18 < height else max(0, y1 - 18)
            draw_label(draw, (x1, label_y), label, font, fill=(150, 220, 255))


def draw_gt_annotations(
    image: Image.Image,
    texts: list[str],
    bboxes: list[Any],
    font: ImageFont.ImageFont,
) -> None:
    draw = ImageDraw.Draw(image)
    width, height = image.size
    for index, raw_bbox in enumerate(bboxes):
        if (
            not isinstance(raw_bbox, list)
            or len(raw_bbox) != 2
            or len(raw_bbox[0]) != 2
            or len(raw_bbox[1]) != 2
        ):
            continue
        x1, y1 = raw_bbox[0]
        x2, y2 = raw_bbox[1]
        x1 = max(0, min(width - 1, int(x1)))
        y1 = max(0, min(height - 1, int(y1)))
        x2 = max(0, min(width - 1, int(x2)))
        y2 = max(0, min(height - 1, int(y2)))
        if x2 <= x1 or y2 <= y1:
            continue
        draw.rectangle((x1, y1, x2, y2), outline=(255, 40, 40), width=2)
        if index < len(texts) and texts[index]:
            label = str(texts[index])
            if len(label) > 24:
                label = label[:21] + "..."
            label_y = y1 - 18 if y1 >= 20 else y1 + 2
            draw_label(draw, (x1, label_y), label, font, fill=(255, 255, 0))


def make_pair_image(
    image_id: str,
    gt_image: Image.Image,
    diffbir_image: Image.Image,
    texts: list[str],
    bboxes: list[Any],
    bridge_annotations: list[dict[str, Any]],
    draw_bboxes: bool,
    font: ImageFont.ImageFont,
    small_font: ImageFont.ImageFont,
) -> Image.Image:
    gt_panel = gt_image.copy().convert("RGB")
    diffbir_panel = diffbir_image.copy().convert("RGB")
    if diffbir_panel.size != gt_panel.size:
        diffbir_panel = diffbir_panel.resize(gt_panel.size, Image.Resampling.LANCZOS)

    if draw_bboxes:
        draw_gt_annotations(gt_panel, texts, bboxes, small_font)
        if bridge_annotations:
            draw_bridge_annotations(diffbir_panel, bridge_annotations, small_font)

    gap = 12
    header_h = 46
    label_h = 28
    width = gt_panel.width * 2 + gap
    height = gt_panel.height + header_h + label_h
    canvas = Image.new("RGB", (width, height), (22, 22, 22))
    draw = ImageDraw.Draw(canvas)
    draw.text((10, 8), image_id, font=font, fill=(255, 255, 255))
    draw.text(
        (10, header_h),
        "GT + annotation (red)",
        font=small_font,
        fill=(255, 120, 120),
    )
    right_label = (
        f"DiffBIR + Bridge prediction (blue, n={len(bridge_annotations)})"
        if bridge_annotations
        else "DiffBIR inference"
    )
    draw.text(
        (gt_panel.width + gap + 10, header_h),
        right_label,
        font=small_font,
        fill=(150, 220, 255) if bridge_annotations else (255, 255, 255),
    )
    canvas.paste(gt_panel, (0, header_h + label_h))
    canvas.paste(diffbir_panel, (gt_panel.width + gap, header_h + label_h))
    return canvas


def make_contact_sheets(
    chunk_dir: Path,
    pairs: list[Path],
    cols: int,
    rows: int,
    thumb_width: int,
    font: ImageFont.ImageFont,
) -> None:
    if not pairs:
        return
    sheet_dir = chunk_dir / "contact_sheets"
    sheet_dir.mkdir(parents=True, exist_ok=True)
    per_page = cols * rows
    for page_index in range(math.ceil(len(pairs) / per_page)):
        page_paths = pairs[page_index * per_page : (page_index + 1) * per_page]
        thumbs: list[Image.Image] = []
        for path in page_paths:
            with Image.open(path) as image:
                image = image.convert("RGB")
                thumb_height = max(1, round(image.height * (thumb_width / image.width)))
                thumbs.append(image.resize((thumb_width, thumb_height), Image.Resampling.LANCZOS))
        if not thumbs:
            continue
        cell_w = thumb_width
        cell_h = max(thumb.height for thumb in thumbs) + 24
        pad = 10
        sheet = Image.new(
            "RGB",
            (cols * cell_w + (cols + 1) * pad, rows * cell_h + (rows + 1) * pad),
            (30, 30, 30),
        )
        draw = ImageDraw.Draw(sheet)
        for idx, thumb in enumerate(thumbs):
            r = idx // cols
            c = idx % cols
            x = pad + c * (cell_w + pad)
            y = pad + r * (cell_h + pad)
            label = page_paths[idx].stem.replace("_gt_diffbir", "")
            if len(label) > 42:
                label = label[:39] + "..."
            draw.text((x, y), label, font=font, fill=(255, 255, 255))
            sheet.paste(thumb, (x, y + 24))
        sheet.save(sheet_dir / f"overview_page_{page_index:03d}.png")


def process_chunk(
    chunk: str,
    results_root: Path,
    output_root: Path,
    gt_rows: dict[str, dict[str, Any]],
    bridge_rows: dict[str, list[dict[str, Any]]],
    max_images: int | None,
    draw_bboxes: bool,
    sheet_cols: int,
    sheet_rows: int,
    thumb_width: int,
) -> tuple[int, int]:
    chunk_input_dir = results_root / chunk
    chunk_output_dir = output_root / chunk
    chunk_output_dir.mkdir(parents=True, exist_ok=True)
    font = load_font(18)
    small_font = load_font(13)

    image_paths = sorted(chunk_input_dir.glob("*.png"))
    if max_images is not None:
        image_paths = image_paths[:max_images]

    written: list[Path] = []
    missing_gt = 0
    manifest_rows: list[dict[str, Any]] = []
    for diffbir_path in image_paths:
        image_id = diffbir_path.stem
        gt_row = gt_rows.get(image_id)
        if gt_row is None:
            missing_gt += 1
            continue
        gt_image = image_from_hf_value(gt_row["hq_img"])
        diffbir_image = Image.open(diffbir_path).convert("RGB")
        bridge_anns = bridge_rows.get(image_id, [])
        pair = make_pair_image(
            image_id=image_id,
            gt_image=gt_image,
            diffbir_image=diffbir_image,
            texts=list(gt_row.get("text") or []),
            bboxes=list(gt_row.get("bbox") or []),
            bridge_annotations=bridge_anns,
            draw_bboxes=draw_bboxes,
            font=font,
            small_font=small_font,
        )
        output_path = chunk_output_dir / f"{image_id}_gt_diffbir.png"
        pair.save(output_path)
        written.append(output_path)
        manifest_rows.append(
            {
                "chunk": chunk,
                "image_id": image_id,
                "diffbir_path": str(diffbir_path),
                "visualization_path": str(output_path),
                "num_gt_text": len(gt_row.get("text") or []),
                "num_gt_bbox": len(gt_row.get("bbox") or []),
                "num_bridge_pred": len(bridge_anns),
            }
        )

    if manifest_rows:
        with (chunk_output_dir / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(manifest_rows[0].keys()))
            writer.writeheader()
            writer.writerows(manifest_rows)

    make_contact_sheets(
        chunk_output_dir,
        written,
        cols=sheet_cols,
        rows=sheet_rows,
        thumb_width=thumb_width,
        font=small_font,
    )
    return len(written), missing_gt


def main() -> None:
    args = parse_args()
    output_root = args.output_dir or (args.results_root / "GT_DiffBIR_inference")
    gt_rows = load_gt_rows(args.parquet)
    total_written = 0
    total_missing = 0
    for chunk in args.chunks:
        bridge_rows: dict[str, list[dict[str, Any]]] = {}
        if args.bridge_json_fmt:
            bridge_path = Path(
                args.bridge_json_fmt.format(
                    results_root=str(args.results_root),
                    chunk=chunk,
                )
            )
            bridge_rows = load_bridge_predictions(bridge_path, args.bridge_score_threshold)
            if bridge_rows:
                print(f"{chunk}: loaded Bridge predictions for {len(bridge_rows)} images from {bridge_path}")
            else:
                print(f"{chunk}: no Bridge predictions found at {bridge_path}")

        written, missing = process_chunk(
            chunk=chunk,
            results_root=args.results_root,
            output_root=output_root,
            gt_rows=gt_rows,
            bridge_rows=bridge_rows,
            max_images=args.max_images_per_chunk,
            draw_bboxes=not args.no_bboxes,
            sheet_cols=args.contact_sheet_cols,
            sheet_rows=args.contact_sheet_rows,
            thumb_width=args.thumb_width,
        )
        total_written += written
        total_missing += missing
        print(f"{chunk}: wrote {written} visualizations, missing GT {missing}")
    print(f"Output: {output_root}")
    print(f"Total written: {total_written}, total missing GT: {total_missing}")


if __name__ == "__main__":
    main()
