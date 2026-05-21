#!/usr/bin/env python3
"""Create GT/LQ/TAIR/DiffBIR comparison sheets for SA-Text chunks."""

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


RUN_ROOT = Path(
    "/scratch2/james2602/TextAwareIR/Results/Baselines/TAIR/"
    "sa_text_test_lv2_chunks_0_1_stage3"
)
GT_PARQUET = Path(
    "/scratch2/james2602/TextAwareIR/Dataset/SA-Text-test/data/"
    "test-00000-of-00001.parquet"
)
DIFFBIR_ROOT = Path(
    "/scratch2/james2602/TextAwareIR/Results/Baselines/DiffBIR/"
    "sa_text_test_lv2_a6000"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parquet", type=Path, default=GT_PARQUET)
    parser.add_argument("--run-root", type=Path, default=RUN_ROOT)
    parser.add_argument("--lq-root", type=Path, default=None)
    parser.add_argument("--tair-root", type=Path, default=None)
    parser.add_argument("--diffbir-root", type=Path, default=DIFFBIR_ROOT)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--chunks", nargs="+", default=["chunk_0", "chunk_1"])
    parser.add_argument(
        "--tair-json-fmt",
        type=str,
        default="{run_root}/text_annotations/tair/{chunk}/pipeline/bridge_filtered.json",
    )
    parser.add_argument(
        "--diffbir-json-fmt",
        type=str,
        default="{diffbir_root}/text_annotations/{chunk}/pipeline/bridge_filtered.json",
    )
    parser.add_argument("--max-images-per-chunk", type=int, default=None)
    parser.add_argument("--line-width", type=int, default=5)
    parser.add_argument("--contact-sheet-cols", type=int, default=2)
    parser.add_argument("--contact-sheet-rows", type=int, default=4)
    parser.add_argument("--thumb-width", type=int, default=560)
    return parser.parse_args()


def import_pyarrow() -> Any:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise SystemExit(
            "pyarrow is required to read SA-Text-test parquet files. "
            "Use the diffbir env or install pyarrow."
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
    return {safe_stem(str(row["id"])): row for row in table.to_pylist()}


def load_font(size: int) -> ImageFont.ImageFont:
    for candidate in (
        "/usr/share/fonts/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
    ):
        path = Path(candidate)
        if path.is_file():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def annotation_text(annotation: dict[str, Any]) -> str:
    for key in ("VLM", "text", "rec", "OVIS", "Qwen"):
        value = annotation.get(key)
        if value is not None:
            return str(value)
    return ""


def bbox_from_polygon(raw_polygon: Any) -> tuple[float, float, float, float] | None:
    if not raw_polygon:
        return None
    points = raw_polygon
    if points and not isinstance(points[0], (list, tuple)):
        if len(points) % 2:
            return None
        points = list(zip(points[0::2], points[1::2]))
    xs = [float(point[0]) for point in points]
    ys = [float(point[1]) for point in points]
    return min(xs), min(ys), max(xs), max(ys)


def load_prediction_annotations(json_path: Path) -> dict[str, list[dict[str, Any]]]:
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
        bbox = ann.get("bbox")
        if (not isinstance(bbox, (list, tuple)) or len(bbox) != 4) and ann.get("polygon"):
            bbox = bbox_from_polygon(ann.get("polygon"))
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            continue
        grouped[safe_stem(file_name)].append(
            {
                "bbox": tuple(float(v) for v in bbox),
                "text": annotation_text(ann),
            }
        )
    return grouped


def draw_label(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    font: ImageFont.ImageFont,
    fill: tuple[int, int, int],
) -> None:
    if not text:
        return
    x, y = xy
    label = text if len(text) <= 24 else text[:21] + "..."
    bbox = draw.textbbox((x, y), label, font=font)
    pad = 4
    draw.rectangle(
        (bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad),
        fill=(0, 0, 0),
    )
    draw.text((x, y), label, font=font, fill=fill)


def draw_boxes(
    image: Image.Image,
    annotations: list[dict[str, Any]],
    source_size: tuple[int, int],
    width: int,
    font: ImageFont.ImageFont,
) -> None:
    draw = ImageDraw.Draw(image)
    dst_w, dst_h = image.size
    src_w, src_h = source_size
    sx = dst_w / max(src_w, 1)
    sy = dst_h / max(src_h, 1)
    red = (255, 0, 0)
    for ann in annotations:
        x1, y1, x2, y2 = ann["bbox"]
        x1 = int(round(x1 * sx))
        x2 = int(round(x2 * sx))
        y1 = int(round(y1 * sy))
        y2 = int(round(y2 * sy))
        x1 = max(0, min(dst_w - 1, x1))
        x2 = max(0, min(dst_w - 1, x2))
        y1 = max(0, min(dst_h - 1, y1))
        y2 = max(0, min(dst_h - 1, y2))
        if x2 <= x1 or y2 <= y1:
            continue
        draw.rectangle((x1, y1, x2, y2), outline=red, width=width)
        label_y = y1 - 22 if y1 >= 24 else y2 + 4
        draw_label(draw, (x1, label_y), ann.get("text", ""), font, fill=(255, 220, 220))


def gt_annotations(texts: list[str], bboxes: list[Any]) -> list[dict[str, Any]]:
    anns: list[dict[str, Any]] = []
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
        anns.append(
            {
                "bbox": (float(x1), float(y1), float(x2), float(y2)),
                "text": str(texts[index]) if index < len(texts) else "",
            }
        )
    return anns


def make_panel(
    title: str,
    image: Image.Image,
    target_size: tuple[int, int],
    annotations: list[dict[str, Any]],
    line_width: int,
    title_font: ImageFont.ImageFont,
    label_font: ImageFont.ImageFont,
) -> Image.Image:
    source_size = image.size
    panel_image = image.convert("RGB").resize(target_size, Image.Resampling.LANCZOS)
    if annotations:
        draw_boxes(panel_image, annotations, source_size, line_width, label_font)

    header_h = 34
    panel = Image.new("RGB", (target_size[0], target_size[1] + header_h), (22, 22, 22))
    draw = ImageDraw.Draw(panel)
    draw.text((10, 8), title, font=title_font, fill=(255, 255, 255))
    panel.paste(panel_image, (0, header_h))
    return panel


def make_row(
    image_id: str,
    gt_image: Image.Image,
    lq_image: Image.Image,
    tair_image: Image.Image,
    diffbir_image: Image.Image,
    gt_anns: list[dict[str, Any]],
    tair_anns: list[dict[str, Any]],
    diffbir_anns: list[dict[str, Any]],
    line_width: int,
    title_font: ImageFont.ImageFont,
    label_font: ImageFont.ImageFont,
) -> Image.Image:
    target_size = gt_image.size
    panels = [
        make_panel("GT annotation", gt_image, target_size, gt_anns, line_width, title_font, label_font),
        make_panel("LQ2 input", lq_image, target_size, [], line_width, title_font, label_font),
        make_panel("TAIR annotation", tair_image, target_size, tair_anns, line_width, title_font, label_font),
        make_panel("DiffBIR annotation", diffbir_image, target_size, diffbir_anns, line_width, title_font, label_font),
    ]
    gap = 12
    top_h = 38
    width = sum(panel.width for panel in panels) + gap * (len(panels) - 1)
    height = max(panel.height for panel in panels) + top_h
    canvas = Image.new("RGB", (width, height), (18, 18, 18))
    draw = ImageDraw.Draw(canvas)
    draw.text((12, 8), image_id, font=title_font, fill=(255, 255, 255))
    x = 0
    for panel in panels:
        canvas.paste(panel, (x, top_h))
        x += panel.width + gap
    return canvas


def make_contact_sheets(
    chunk_dir: Path,
    rows: list[Path],
    cols: int,
    sheet_rows: int,
    thumb_width: int,
    font: ImageFont.ImageFont,
) -> None:
    if not rows:
        return
    sheet_dir = chunk_dir / "contact_sheets"
    sheet_dir.mkdir(parents=True, exist_ok=True)
    per_page = cols * sheet_rows
    for page_index in range(math.ceil(len(rows) / per_page)):
        page_paths = rows[page_index * per_page : (page_index + 1) * per_page]
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
            (cols * cell_w + (cols + 1) * pad, sheet_rows * cell_h + (sheet_rows + 1) * pad),
            (30, 30, 30),
        )
        draw = ImageDraw.Draw(sheet)
        for idx, thumb in enumerate(thumbs):
            row = idx // cols
            col = idx % cols
            x = pad + col * (cell_w + pad)
            y = pad + row * (cell_h + pad)
            label = page_paths[idx].stem.replace("_gt_lq_tair_diffbir", "")
            if len(label) > 48:
                label = label[:45] + "..."
            draw.text((x, y), label, font=font, fill=(255, 255, 255))
            sheet.paste(thumb, (x, y + 24))
        sheet.save(sheet_dir / f"overview_page_{page_index:03d}.png")


def format_annotation_path(fmt: str, run_root: Path, diffbir_root: Path, chunk: str) -> Path:
    return Path(
        fmt.format(
            run_root=str(run_root),
            diffbir_root=str(diffbir_root),
            chunk=chunk,
        )
    )


def process_chunk(
    chunk: str,
    args: argparse.Namespace,
    gt_rows: dict[str, dict[str, Any]],
    lq_root: Path,
    tair_root: Path,
    output_root: Path,
) -> tuple[int, int]:
    chunk_lq_dir = lq_root / chunk
    chunk_tair_dir = tair_root / chunk
    chunk_diffbir_dir = args.diffbir_root / chunk
    chunk_output_dir = output_root / chunk
    chunk_output_dir.mkdir(parents=True, exist_ok=True)

    tair_ann_path = format_annotation_path(args.tair_json_fmt, args.run_root, args.diffbir_root, chunk)
    diffbir_ann_path = format_annotation_path(args.diffbir_json_fmt, args.run_root, args.diffbir_root, chunk)
    tair_annotations = load_prediction_annotations(tair_ann_path)
    diffbir_annotations = load_prediction_annotations(diffbir_ann_path)

    title_font = load_font(18)
    label_font = load_font(13)

    image_paths = sorted(chunk_lq_dir.glob("*.png"))
    if args.max_images_per_chunk is not None:
        image_paths = image_paths[: args.max_images_per_chunk]

    written_paths: list[Path] = []
    manifest_rows: list[dict[str, Any]] = []
    skipped = 0
    for lq_path in image_paths:
        image_id = lq_path.stem
        gt_row = gt_rows.get(image_id)
        tair_path = chunk_tair_dir / f"{image_id}.png"
        diffbir_path = chunk_diffbir_dir / f"{image_id}.png"
        if gt_row is None or not tair_path.is_file() or not diffbir_path.is_file():
            skipped += 1
            continue

        gt_image = image_from_hf_value(gt_row["hq_img"])
        with Image.open(lq_path) as lq_image_raw, Image.open(tair_path) as tair_image_raw, Image.open(diffbir_path) as diffbir_image_raw:
            row_image = make_row(
                image_id=image_id,
                gt_image=gt_image,
                lq_image=lq_image_raw.convert("RGB"),
                tair_image=tair_image_raw.convert("RGB"),
                diffbir_image=diffbir_image_raw.convert("RGB"),
                gt_anns=gt_annotations(list(gt_row.get("text") or []), list(gt_row.get("bbox") or [])),
                tair_anns=tair_annotations.get(image_id, []),
                diffbir_anns=diffbir_annotations.get(image_id, []),
                line_width=args.line_width,
                title_font=title_font,
                label_font=label_font,
            )

        output_path = chunk_output_dir / f"{image_id}_gt_lq_tair_diffbir.png"
        row_image.save(output_path)
        written_paths.append(output_path)
        manifest_rows.append(
            {
                "chunk": chunk,
                "image_id": image_id,
                "lq_path": str(lq_path),
                "tair_path": str(tair_path),
                "diffbir_path": str(diffbir_path),
                "visualization_path": str(output_path),
                "num_gt_annotations": len(gt_row.get("bbox") or []),
                "num_tair_annotations": len(tair_annotations.get(image_id, [])),
                "num_diffbir_annotations": len(diffbir_annotations.get(image_id, [])),
            }
        )

    if manifest_rows:
        with (chunk_output_dir / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(manifest_rows[0].keys()))
            writer.writeheader()
            writer.writerows(manifest_rows)

    make_contact_sheets(
        chunk_output_dir,
        written_paths,
        cols=args.contact_sheet_cols,
        sheet_rows=args.contact_sheet_rows,
        thumb_width=args.thumb_width,
        font=label_font,
    )
    return len(written_paths), skipped


def main() -> None:
    args = parse_args()
    lq_root = args.lq_root or (args.run_root / "inputs" / "lq2")
    tair_root = args.tair_root or (args.run_root / "tair_hq")
    output_root = args.output_dir or (args.run_root / "visualizations" / "gt_lq_tair_diffbir")
    output_root.mkdir(parents=True, exist_ok=True)

    gt_rows = load_gt_rows(args.parquet)
    total_written = 0
    total_skipped = 0
    for chunk in args.chunks:
        written, skipped = process_chunk(
            chunk=chunk,
            args=args,
            gt_rows=gt_rows,
            lq_root=lq_root,
            tair_root=tair_root,
            output_root=output_root,
        )
        total_written += written
        total_skipped += skipped
        print(f"{chunk}: wrote {written} visualizations, skipped {skipped}")

    print(f"Output: {output_root}")
    print(f"Total written: {total_written}, total skipped: {total_skipped}")


if __name__ == "__main__":
    main()
