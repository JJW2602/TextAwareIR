#!/usr/bin/env python3
from __future__ import annotations

import argparse
import io
import json
import math
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[2]
TAIR_ROOT = REPO_ROOT / "TAIR"
sys.path.insert(0, str(TAIR_ROOT))

from terediff.dataset.batch_transform import RealESRGANBatchTransform  # noqa: E402
from terediff.dataset.degradation import circular_lowpass_kernel, random_mixed_kernels  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=REPO_ROOT / "TAIRL/train_data/sa_text_train_10000.jsonl",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "Dataset/SA-Text-lv2-10000",
    )
    parser.add_argument("--count", type=int, default=10_000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--shard-size", type=int, default=1_000)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--lq-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=231)
    parser.add_argument("--device", type=str, default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def image_from_hf_value(value: Any) -> Image.Image:
    if isinstance(value, dict):
        if value.get("bytes") is not None:
            return Image.open(io.BytesIO(value["bytes"])).convert("RGB")
        if value.get("path"):
            return Image.open(value["path"]).convert("RGB")
    if isinstance(value, (bytes, bytearray)):
        return Image.open(io.BytesIO(value)).convert("RGB")
    raise ValueError(f"Unsupported image payload type: {type(value)!r}")


def encode_png(image: Image.Image) -> dict[str, bytes | None]:
    with io.BytesIO() as handle:
        image.save(handle, format="PNG")
        return {"bytes": handle.getvalue(), "path": None}


def tensor_to_png_payload(image: torch.Tensor) -> dict[str, bytes | None]:
    arr = (
        image.detach()
        .float()
        .clamp(0, 1)
        .cpu()
        .mul(255)
        .byte()
        .permute(1, 2, 0)
        .numpy()
    )
    return encode_png(Image.fromarray(arr))


def pil_to_tensor(image: Image.Image, image_size: int) -> torch.Tensor:
    if image.size != (image_size, image_size):
        image = image.resize((image_size, image_size), Image.BICUBIC)
    arr = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


def make_transform(batch_size: int) -> RealESRGANBatchTransform:
    return RealESRGANBatchTransform(
        use_sharpener=True,
        queue_size=max(batch_size, 1),
        resize_prob=[0.2, 0.7, 0.1],
        resize_range=[0.15, 1.5],
        gaussian_noise_prob=0.5,
        noise_range=[1, 30],
        poisson_scale_range=[0.05, 3],
        gray_noise_prob=0.4,
        jpeg_range=[30, 95],
        stage2_scale=4,
        second_blur_prob=0.8,
        resize_prob2=[0.3, 0.4, 0.3],
        resize_range2=[0.3, 1.2],
        gaussian_noise_prob2=0.5,
        noise_range2=[1, 25],
        poisson_scale_range2=[0.05, 2.5],
        gray_noise_prob2=0.4,
        jpeg_range2=[30, 95],
    )


def make_kernel_batch(batch_size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    kernel_range = [2 * v + 1 for v in range(3, 11)]
    pulse_tensor = torch.zeros(21, 21).float()
    pulse_tensor[10, 10] = 1

    kernels1: list[torch.Tensor] = []
    kernels2: list[torch.Tensor] = []
    sinc_kernels: list[torch.Tensor] = []
    for _ in range(batch_size):
        kernel_size = random.choice(kernel_range)
        if np.random.uniform() < 0.1:
            omega_c = np.random.uniform(np.pi / 3, np.pi) if kernel_size < 13 else np.random.uniform(np.pi / 5, np.pi)
            kernel = circular_lowpass_kernel(omega_c, kernel_size, pad_to=False)
        else:
            kernel = random_mixed_kernels(
                ["iso", "aniso", "generalized_iso", "generalized_aniso", "plateau_iso", "plateau_aniso"],
                [0.45, 0.25, 0.12, 0.03, 0.12, 0.03],
                kernel_size,
                [0.2, 3],
                [0.2, 3],
                [-math.pi, math.pi],
                [0.5, 4],
                [1, 2],
                noise_range=None,
            )
        pad_size = (21 - kernel_size) // 2
        kernels1.append(torch.FloatTensor(np.pad(kernel, ((pad_size, pad_size), (pad_size, pad_size)))))

        kernel_size = random.choice(kernel_range)
        if np.random.uniform() < 0.1:
            omega_c = np.random.uniform(np.pi / 3, np.pi) if kernel_size < 13 else np.random.uniform(np.pi / 5, np.pi)
            kernel2 = circular_lowpass_kernel(omega_c, kernel_size, pad_to=False)
        else:
            kernel2 = random_mixed_kernels(
                ["iso", "aniso", "generalized_iso", "generalized_aniso", "plateau_iso", "plateau_aniso"],
                [0.45, 0.25, 0.12, 0.03, 0.12, 0.03],
                kernel_size,
                [0.2, 1.5],
                [0.2, 1.5],
                [-math.pi, math.pi],
                [0.5, 4],
                [1, 2],
                noise_range=None,
            )
        pad_size = (21 - kernel_size) // 2
        kernels2.append(torch.FloatTensor(np.pad(kernel2, ((pad_size, pad_size), (pad_size, pad_size)))))

        if np.random.uniform() < 0.8:
            kernel_size = random.choice(kernel_range)
            omega_c = np.random.uniform(np.pi / 3, np.pi)
            sinc_kernels.append(torch.FloatTensor(circular_lowpass_kernel(omega_c, kernel_size, pad_to=21)))
        else:
            sinc_kernels.append(pulse_tensor.clone())

    return (
        torch.stack(kernels1).to(device),
        torch.stack(kernels2).to(device),
        torch.stack(sinc_kernels).to(device),
    )


def load_manifest(path: Path, count: int) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                entries.append(json.loads(line))
            if len(entries) >= count:
                break
    if len(entries) < count:
        raise RuntimeError(f"Requested {count} samples, but manifest only has {len(entries)}.")
    return entries


def load_rows(entries: list[dict[str, Any]], columns: list[str]) -> list[dict[str, Any]]:
    import pyarrow.parquet as pq

    cache: dict[Path, list[dict[str, Any]]] = {}
    rows: list[dict[str, Any]] = []
    for entry in entries:
        parquet_path = Path(entry["parquet"])
        if parquet_path not in cache:
            cache[parquet_path] = pq.read_table(parquet_path, columns=columns).to_pylist()
        rows.append(cache[parquet_path][int(entry["row_index"])])
    return rows


def write_shard(rows: list[dict[str, Any]], output_path: Path) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    output_path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, output_path, compression="zstd")


@torch.no_grad()
def degrade_batch(
    transform: RealESRGANBatchTransform,
    hq: torch.Tensor,
    ids: list[str],
    device: torch.device,
) -> torch.Tensor:
    batch_size = hq.shape[0]
    kernel1, kernel2, sinc_kernel = make_kernel_batch(batch_size, device)
    batch = {
        "hq": hq.to(device),
        "kernel1": kernel1,
        "kernel2": kernel2,
        "sinc_kernel": sinc_kernel,
        "prompt": ["" for _ in ids],
        "text": [[] for _ in ids],
        "bbox": [[] for _ in ids],
        "poly": [[] for _ in ids],
        "text_enc": [[] for _ in ids],
        "img_name": ids,
    }
    _, lq, *_ = transform(batch)
    return lq.permute(0, 3, 1, 2).contiguous().cpu()


def main() -> None:
    args = parse_args()
    if args.count <= 0:
        raise SystemExit("--count must be positive.")
    if args.batch_size <= 0:
        raise SystemExit("--batch-size must be positive.")
    if args.shard_size <= 0:
        raise SystemExit("--shard-size must be positive.")
    if args.lq_size <= 0:
        raise SystemExit("--lq-size must be positive.")

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True

    seed_everything(args.seed)
    entries = load_manifest(args.manifest, args.count)
    source_rows = load_rows(entries, columns=["id", "image", "text", "bbox", "poly"])
    transform = make_transform(args.batch_size)

    data_dir = args.output_dir / "data"
    if data_dir.exists() and any(data_dir.glob("train-*.parquet")) and not args.overwrite:
        raise FileExistsError(f"Output shards already exist under {data_dir}; pass --overwrite to replace them.")
    if args.overwrite and data_dir.exists():
        for old_path in data_dir.glob("train-*.parquet"):
            old_path.unlink()

    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_manifest": str(args.manifest),
        "count": args.count,
        "level": 2,
        "image_size": args.image_size,
        "lq_size": args.lq_size,
        "device": str(device),
        "transform": "TAIR.terediff.dataset.batch_transform.RealESRGANBatchTransform",
        "stage2_scale": 4,
        "output_dir": str(args.output_dir),
    }

    shard_rows: list[dict[str, Any]] = []
    shard_index = 0
    for batch_start in range(0, args.count, args.batch_size):
        batch_end = min(batch_start + args.batch_size, args.count)
        batch_rows = source_rows[batch_start:batch_end]
        batch_entries = entries[batch_start:batch_end]
        ids = [str(entry["id"]) for entry in batch_entries]

        hq_images = [image_from_hf_value(row["image"]) for row in batch_rows]
        hq_tensors = torch.stack([pil_to_tensor(image, args.image_size) for image in hq_images], dim=0)
        lq_tensors = degrade_batch(transform, hq_tensors, ids, device)
        if lq_tensors.shape[-2:] != (args.lq_size, args.lq_size):
            lq_tensors = F.interpolate(
                lq_tensors,
                size=(args.lq_size, args.lq_size),
                mode="bicubic",
                antialias=True,
            ).clamp(0, 1)

        for entry, source_row, hq_tensor, lq_tensor in zip(batch_entries, batch_rows, hq_tensors, lq_tensors):
            shard_rows.append(
                {
                    "id": str(entry["id"]),
                    "hq_img": tensor_to_png_payload(hq_tensor),
                    "lq_img_lv2": tensor_to_png_payload(lq_tensor),
                    "text": source_row.get("text") or [],
                    "bbox": source_row.get("bbox") or [],
                    "poly": source_row.get("poly") or [],
                    "source_parquet": str(entry["parquet"]),
                    "source_row_index": int(entry["row_index"]),
                    "sample_index": int(entry["sample_index"]),
                }
            )

        while len(shard_rows) >= args.shard_size:
            rows_to_write = shard_rows[: args.shard_size]
            shard_rows = shard_rows[args.shard_size :]
            shard_path = data_dir / f"train-{shard_index:05d}-of-unknown.parquet"
            write_shard(rows_to_write, shard_path)
            print(f"wrote {shard_path} ({len(rows_to_write)} rows)", flush=True)
            shard_index += 1

        print(f"processed {batch_end}/{args.count}", flush=True)

    if shard_rows:
        shard_path = data_dir / f"train-{shard_index:05d}-of-unknown.parquet"
        write_shard(shard_rows, shard_path)
        print(f"wrote {shard_path} ({len(shard_rows)} rows)", flush=True)
        shard_index += 1

    old_paths = sorted(data_dir.glob("train-*-of-unknown.parquet"))
    final_paths: list[Path] = []
    for index, old_path in enumerate(old_paths):
        final_path = data_dir / f"train-{index:05d}-of-{shard_index:05d}.parquet"
        if final_path.exists():
            final_path.unlink()
        old_path.rename(final_path)
        final_paths.append(final_path)

    manifest_path = args.output_dir / "sa_text_train_10000_lv2.jsonl"
    with manifest_path.open("w", encoding="utf-8") as handle:
        running_index = 0
        for shard_path in final_paths:
            import pyarrow.parquet as pq

            ids = pq.read_table(shard_path, columns=["id"]).column("id").to_pylist()
            for row_index, image_id in enumerate(ids):
                handle.write(
                    json.dumps(
                        {
                            "sample_index": running_index,
                            "id": str(image_id),
                            "parquet": str(shard_path),
                            "row_index": row_index,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                running_index += 1

    summary["shards"] = [str(path) for path in final_paths]
    summary["manifest"] = str(manifest_path)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (args.output_dir / "README.md").write_text(
        "# SA-Text LV2 10k\n\n"
        "Generated from SA-Text train images using "
        "`TAIR/terediff/dataset/batch_transform.py::RealESRGANBatchTransform`.\n\n"
        "`hq_img` is stored at 512x512. `lq_img_lv2` is stored at 128x128 to match "
        "the raw SA-Text-test lv2 image size.\n\n"
        "Columns: `id`, `hq_img`, `lq_img_lv2`, `text`, `bbox`, `poly`, "
        "`source_parquet`, `source_row_index`, `sample_index`.\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
