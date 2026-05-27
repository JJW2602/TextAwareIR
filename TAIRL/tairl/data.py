from __future__ import annotations

import io
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF


@dataclass(frozen=True)
class TextInstance:
    text: str
    bbox: tuple[float, float, float, float]


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


def _scale_bbox(raw_bbox: Any, sx: float, sy: float) -> tuple[float, float, float, float] | None:
    if not (
        isinstance(raw_bbox, (list, tuple))
        and len(raw_bbox) == 2
        and len(raw_bbox[0]) == 2
        and len(raw_bbox[1]) == 2
    ):
        return None
    x1, y1 = raw_bbox[0]
    x2, y2 = raw_bbox[1]
    return float(x1) * sx, float(y1) * sy, float(x2) * sx, float(y2) * sy


def _build_prompt(texts: list[str], mode: str, fixed_prompt: str) -> str:
    if mode == "empty":
        return ""
    if mode == "fixed":
        return fixed_prompt
    if mode == "gt_text":
        clean = [f'"{t}"' for t in texts if str(t).strip()]
        if clean:
            return (
                "A realistic image where the text "
                + ", ".join(clean[:8])
                + " appears clearly and sharply."
            )
        return fixed_prompt
    raise ValueError(f"Unsupported prompt mode: {mode}")


class SATextParquetDataset(Dataset):
    """Small SA-Text parquet loader for online reward fine-tuning."""

    def __init__(
        self,
        parquet_path: str | Path | None,
        manifest_path: str | Path | None,
        level: int,
        start: int,
        max_samples: int | None,
        image_size: int,
        prompt_mode: str,
        fixed_prompt: str,
        negative_prompt: str,
    ) -> None:
        try:
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise RuntimeError("pyarrow is required to read SA-Text parquet files.") from exc

        self._pq = pq
        self._shard_cache: dict[Path, list[dict[str, Any]]] = {}
        self.parquet_path = None if parquet_path is None else Path(parquet_path)
        self.level = level
        self.image_size = image_size
        self.prompt_mode = prompt_mode
        self.fixed_prompt = fixed_prompt
        self.negative_prompt = negative_prompt
        self.source_description = str(manifest_path or parquet_path)

        if manifest_path:
            if start != 0:
                raise ValueError("start must be 0 when data.manifest is used.")
            self.rows = self._load_manifest_rows(
                pq=pq,
                manifest_path=Path(manifest_path),
                level=level,
                max_samples=max_samples,
            )
        else:
            if parquet_path is None:
                raise ValueError("Either parquet_path or manifest_path must be provided.")
            self.parquet_path = Path(parquet_path)
            self.rows = self._load_parquet_rows(
                pq=pq,
                parquet_path=self.parquet_path,
                level=level,
                start=start,
                max_samples=max_samples,
            )

    @staticmethod
    def _image_columns(pq: Any, parquet_path: Path, level: int) -> tuple[str, str, list[str]]:
        schema_names = set(pq.read_schema(parquet_path).names)
        preferred_lq = f"lq_img_lv{level}"
        if preferred_lq in schema_names:
            lq_column = preferred_lq
        elif "image" in schema_names:
            lq_column = "image"
        elif "hq_img" in schema_names:
            lq_column = "hq_img"
        else:
            raise ValueError(f"No usable image column found in {parquet_path}")

        if "hq_img" in schema_names:
            hq_column = "hq_img"
        elif "image" in schema_names:
            hq_column = "image"
        else:
            hq_column = lq_column

        columns = []
        for column in ("id", lq_column, hq_column, "text", "bbox"):
            if column not in columns:
                columns.append(column)
        return lq_column, hq_column, columns

    @classmethod
    def _load_parquet_rows(
        cls,
        pq: Any,
        parquet_path: Path,
        level: int,
        start: int,
        max_samples: int | None,
    ) -> list[dict[str, Any]]:
        lq_column, hq_column, columns = cls._image_columns(pq, parquet_path, level)
        table = pq.read_table(parquet_path, columns=columns)
        if start < 0 or start >= table.num_rows:
            raise ValueError(f"start={start} is outside parquet with {table.num_rows} rows")
        count = table.num_rows - start if max_samples is None else max_samples
        rows = table.slice(start, min(count, table.num_rows - start)).to_pylist()
        for row in rows:
            row["_lq_column"] = lq_column
            row["_hq_column"] = hq_column
        return rows

    @classmethod
    def _load_manifest_rows(
        cls,
        pq: Any,
        manifest_path: Path,
        level: int,
        max_samples: int | None,
    ) -> list[dict[str, Any]]:
        if not manifest_path.is_file():
            raise FileNotFoundError(f"SA-Text manifest not found: {manifest_path}")

        entries: list[dict[str, Any]] = []
        with manifest_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    entries.append(json.loads(line))
                if max_samples is not None and len(entries) >= max_samples:
                    break

        schema_cache: dict[Path, tuple[str, str, list[str], int]] = {}
        rows: list[dict[str, Any]] = []
        for entry in entries:
            parquet_path = Path(entry["parquet"])
            row_index = int(entry["row_index"])
            if parquet_path not in schema_cache:
                lq_column, hq_column, columns = cls._image_columns(pq, parquet_path, level)
                num_rows = pq.ParquetFile(parquet_path).metadata.num_rows
                schema_cache[parquet_path] = (lq_column, hq_column, columns, num_rows)
            lq_column, hq_column, columns, num_rows = schema_cache[parquet_path]
            if row_index < 0 or row_index >= num_rows:
                raise ValueError(f"row_index={row_index} is outside {parquet_path}")
            rows.append(
                {
                    "id": str(entry["id"]),
                    "_manifest_parquet": str(parquet_path),
                    "_manifest_row_index": row_index,
                    "_lq_column": lq_column,
                    "_hq_column": hq_column,
                    "_columns": columns,
                }
            )
        return rows

    def _materialize_manifest_row(self, row_ref: dict[str, Any]) -> dict[str, Any]:
        parquet_path = Path(row_ref["_manifest_parquet"])
        if parquet_path not in self._shard_cache:
            self._shard_cache[parquet_path] = self._pq.read_table(
                parquet_path,
                columns=row_ref["_columns"],
            ).to_pylist()
        return self._shard_cache[parquet_path][int(row_ref["_manifest_row_index"])]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row_ref = self.rows[index]
        if "_manifest_parquet" in row_ref:
            row = self._materialize_manifest_row(row_ref)
            image_id = safe_stem(str(row_ref["id"]))
            lq_column = row_ref["_lq_column"]
            hq_column = row_ref["_hq_column"]
        else:
            row = row_ref
            image_id = safe_stem(str(row["id"]))
            lq_column = row["_lq_column"]
            hq_column = row["_hq_column"]
        lq = image_from_hf_value(row[lq_column])
        hq = image_from_hf_value(row[hq_column])
        hq_w, hq_h = hq.size
        sx = self.image_size / max(hq_w, 1)
        sy = self.image_size / max(hq_h, 1)

        if lq.size != (self.image_size, self.image_size):
            lq = lq.resize((self.image_size, self.image_size), Image.BICUBIC)
        if hq.size != (self.image_size, self.image_size):
            hq = hq.resize((self.image_size, self.image_size), Image.BICUBIC)

        texts = [str(t) for t in (row.get("text") or [])]
        instances: list[TextInstance] = []
        for text, bbox in zip(texts, row.get("bbox") or []):
            scaled = _scale_bbox(bbox, sx, sy)
            if scaled is not None:
                instances.append(TextInstance(text=text, bbox=scaled))

        return {
            "image_id": image_id,
            "lq": TF.to_tensor(lq),
            "hq": TF.to_tensor(hq),
            "gt_instances": instances,
            "prompt": _build_prompt(texts, self.prompt_mode, self.fixed_prompt),
            "negative_prompt": self.negative_prompt,
        }


def collate_sa_text(batch: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "image_ids": [item["image_id"] for item in batch],
        "lq": torch.stack([item["lq"] for item in batch], dim=0),
        "hq": torch.stack([item["hq"] for item in batch], dim=0),
        "gt_instances": [item["gt_instances"] for item in batch],
        "prompts": [item["prompt"] for item in batch],
        "negative_prompts": [item["negative_prompt"] for item in batch],
    }
