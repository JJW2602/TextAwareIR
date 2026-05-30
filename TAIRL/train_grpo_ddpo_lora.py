#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
import random
import sys
import time
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "DiffBIR"))
sys.path.insert(0, str(REPO_ROOT / "TAIRL"))

from diffbir.model import ControlLDM, Diffusion, SwinIR  # noqa: E402
from diffbir.pipeline import pad_to_multiples_of, wavelet_reconstruction  # noqa: E402
from diffbir.utils.common import instantiate_from_config  # noqa: E402

from tairl.data import SATextParquetDataset, collate_sa_text  # noqa: E402
from tairl.ddpo_sampler import DDPOSpacedSampler  # noqa: E402
from tairl.lora import inject_lora, lora_state_dict, temporary_lora_enabled  # noqa: E402
from tairl.reward import BridgeReward, ConstantReward  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "overrides",
        nargs="*",
        help="OmegaConf dotlist overrides, e.g. train.learning_rate=3e-5",
    )
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def release_cuda_cache(device: torch.device) -> None:
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


def load_config(path: Path, overrides: list[str]) -> Any:
    cfg = OmegaConf.load(path)
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))
    OmegaConf.resolve(cfg)
    return cfg


def maybe_null(value: Any) -> Any | None:
    if value is None:
        return None
    if str(value).lower() in {"", "none", "null"}:
        return None
    return value


def init_wandb(cfg: Any, output_dir: Path, algo: str):
    if not bool(cfg.get("wandb", {}).get("enabled", False)):
        return None
    try:
        import wandb
    except ImportError as exc:
        raise RuntimeError(
            "wandb.enabled=true but wandb is not installed in this Python environment."
        ) from exc

    wandb_cfg = cfg.wandb
    mode = os.environ.get("WANDB_MODE", str(wandb_cfg.mode))
    run_name = maybe_null(wandb_cfg.run_name) or output_dir.name
    group = maybe_null(wandb_cfg.group) or f"{algo}_lora_{cfg.reward.variant}"
    tags = list(wandb_cfg.tags) if wandb_cfg.get("tags") is not None else []
    tags.extend([algo, str(cfg.reward.backend), str(cfg.reward.variant)])

    wandb_dir = output_dir / "wandb"
    wandb_dir.mkdir(parents=True, exist_ok=True)
    run = wandb.init(
        project=str(wandb_cfg.project),
        entity=maybe_null(wandb_cfg.entity),
        name=str(run_name),
        group=str(group),
        tags=tags,
        config=OmegaConf.to_container(cfg, resolve=True),
        dir=str(wandb_dir),
        mode=mode,
    )
    wandb.define_metric("train/step")
    wandb.define_metric("*", step_metric="train/step")
    return run


def tensor_stats(prefix: str, tensor: torch.Tensor) -> dict[str, float]:
    values = tensor.detach().float().cpu()
    if values.numel() == 0:
        return {}
    stats = {
        f"{prefix}/mean": float(values.mean().item()),
        f"{prefix}/min": float(values.min().item()),
        f"{prefix}/max": float(values.max().item()),
    }
    stats[f"{prefix}/std"] = (
        float(values.std(unbiased=False).item()) if values.numel() > 1 else 0.0
    )
    return stats


@torch.no_grad()
def add_psnr_reward_bonus(
    rewards: torch.Tensor,
    images: torch.Tensor,
    targets: torch.Tensor | None,
    cfg: Any,
) -> tuple[torch.Tensor, dict[str, float]]:
    weight = float(cfg.reward.get("psnr_weight", 0.0))
    if weight == 0.0:
        return rewards, {}
    if targets is None:
        raise ValueError("reward.psnr_weight is non-zero, but the batch has no hq target image.")

    cpu_images = images.detach().float().cpu().clamp(0, 1)
    cpu_targets = targets.detach().float().cpu().clamp(0, 1)
    if cpu_targets.shape[0] == 1 and cpu_images.shape[0] > 1:
        cpu_targets = cpu_targets.repeat(cpu_images.shape[0], 1, 1, 1)
    if cpu_targets.shape[0] != cpu_images.shape[0]:
        raise ValueError(
            f"PSNR target batch size {cpu_targets.shape[0]} does not match images {cpu_images.shape[0]}."
        )
    if cpu_targets.shape[2:] != cpu_images.shape[2:]:
        cpu_targets = F.interpolate(
            cpu_targets,
            size=cpu_images.shape[2:],
            mode="bicubic",
            antialias=True,
        ).clamp(0, 1)

    denominator = max(float(cfg.reward.get("psnr_norm_denominator", 40.0)), 1.0e-6)
    mse = (cpu_images - cpu_targets).pow(2).mean(dim=(1, 2, 3)).clamp_min(1.0e-10)
    psnr = 10.0 * torch.log10(1.0 / mse)
    psnr_norm = (psnr / denominator).clamp(0.0, 1.0)
    bonus = psnr_norm * weight
    base_rewards = rewards.detach().float().cpu()
    total_rewards = base_rewards + bonus
    return total_rewards, {
        "reward_ocr_mean": float(base_rewards.mean().item()),
        "reward_psnr_mean": float(psnr.mean().item()),
        "reward_psnr_norm_mean": float(psnr_norm.mean().item()),
        "reward_psnr_bonus_mean": float(bonus.mean().item()),
        "reward_psnr_weight": weight,
        "reward_total_mean": float(total_rewards.mean().item()),
    }


def reward_detail_stats(details: list[Any]) -> dict[str, float]:
    if not details:
        return {}

    def arr(name: str) -> torch.Tensor:
        return torch.tensor([float(getattr(detail, name)) for detail in details])

    stats: dict[str, float] = {}
    for field in (
        "num_gt",
        "num_pred",
        "num_matched",
        "missed",
        "false_positive",
        "matched_mean_reward",
        "final_reward",
        "final_reward_norm",
        "mean_iou_matched",
    ):
        stats.update(tensor_stats(f"reward_detail/{field}", arr(field)))

    num_gt = arr("num_gt").clamp_min(1.0)
    stats["reward_detail/match_rate_gt_mean"] = float((arr("num_matched") / num_gt).mean().item())
    stats["reward_detail/missed_rate_gt_mean"] = float((arr("missed") / num_gt).mean().item())
    stats["reward_detail/false_positive_per_gt_mean"] = float(
        (arr("false_positive") / num_gt).mean().item()
    )
    return stats


@torch.no_grad()
def parameter_l2_norm(params: list[torch.nn.Parameter]) -> float:
    total = 0.0
    for param in params:
        total += float(param.detach().float().pow(2).sum().cpu().item())
    return math.sqrt(total)


def make_progress_bar(enabled: bool, total: int, desc: str, initial: int = 0) -> Any | None:
    if not enabled:
        return None
    try:
        from tqdm.auto import tqdm
    except ImportError:
        print("tqdm is not installed; source-image progress is disabled.")
        return None
    return tqdm(total=total, initial=initial, desc=desc, unit="img", dynamic_ncols=True)


def wandb_image_payload(
    images: torch.Tensor,
    details: list[Any],
    rewards: torch.Tensor,
    max_images: int,
) -> list[Any]:
    import wandb

    payload = []
    cpu_images = images.detach().float().clamp(0, 1).cpu()
    cpu_rewards = rewards.detach().float().cpu()
    for idx, (image, detail) in enumerate(zip(cpu_images[:max_images], details[:max_images])):
        arr = image.mul(255).byte().permute(1, 2, 0).numpy()
        caption = (
            f"{detail.image_id} | reward={float(cpu_rewards[idx]):.4f} | "
            f"matched={detail.num_matched}/{detail.num_gt} | "
            f"missed={detail.missed} | fp={detail.false_positive} | "
            f"text={detail.matched_mean_reward:.3f}"
        )
        payload.append(wandb.Image(arr, caption=caption))
    return payload


def wandb_reward_table(details: list[Any], rewards: torch.Tensor):
    import wandb

    table = wandb.Table(
        columns=[
            "image_id",
            "reward",
            "num_gt",
            "num_pred",
            "num_matched",
            "missed",
            "false_positive",
            "matched_mean_reward",
            "final_reward",
            "final_reward_norm",
            "mean_iou_matched",
        ]
    )
    for detail, reward in zip(details, rewards.detach().float().cpu()):
        table.add_data(
            detail.image_id,
            float(reward.item()),
            detail.num_gt,
            detail.num_pred,
            detail.num_matched,
            detail.missed,
            detail.false_positive,
            detail.matched_mean_reward,
            detail.final_reward,
            detail.final_reward_norm,
            detail.mean_iou_matched,
        )
    return table


def log_wandb_step(
    wandb_run: Any,
    cfg: Any,
    metric: dict[str, Any],
    rewards: torch.Tensor,
    advantages: torch.Tensor,
    details: list[Any],
    images: torch.Tensor,
    device: torch.device,
) -> None:
    if wandb_run is None:
        return
    step = int(metric["step"])
    payload: dict[str, Any] = {
        "train/step": step,
        "train/lr": metric["lr"],
        "train/step_seconds": metric["step_seconds"],
        "train/rollout_seconds": metric["rollout_seconds"],
        "train/reward_seconds": metric["reward_seconds"],
        "train/update_seconds": metric["update_seconds"],
        "train/images_per_update": metric["images_per_update"],
        "train/reward_mean": metric["reward_mean"],
        "train/reward_std": metric["reward_std"],
        "loss/total_loss": metric.get("total_loss", metric["loss"]),
        "loss/grpo_loss": metric.get("grpo_loss", metric["policy_loss"]),
        "loss/kl_loss": metric.get("kl_loss", metric["kl_coef"] * metric["reference_kl"]),
        "loss/reference_kl": metric["reference_kl"],
        "loss/approx_kl_old_new": metric["approx_kl"],
        "loss/clip_frac": metric["clip_frac"],
        "loss/ratio_mean": metric["ratio_mean"],
        "loss/grad_norm": metric["grad_norm"],
        "loss/lora_param_l2_norm": metric["lora_param_l2_norm"],
        "loss/kl_coef": metric["kl_coef"],
        "system/cuda_memory_allocated_gb": (
            torch.cuda.memory_allocated(device) / (1024**3) if device.type == "cuda" else 0.0
        ),
        "system/cuda_memory_reserved_gb": (
            torch.cuda.memory_reserved(device) / (1024**3) if device.type == "cuda" else 0.0
        ),
    }
    for key in ("baseline", "group_reward_mean", "group_reward_std", "group_size"):
        if key in metric:
            payload[f"algorithm/{key}"] = metric[key]
    for key in (
        "reward_ocr_mean",
        "reward_psnr_mean",
        "reward_psnr_norm_mean",
        "reward_psnr_bonus_mean",
        "reward_psnr_weight",
        "reward_total_mean",
    ):
        if key in metric:
            payload[f"reward_extra/{key}"] = metric[key]

    wandb_run.log(payload)


def tensor_to_uint8_hwc(image: torch.Tensor) -> np.ndarray:
    return (
        image.detach()
        .float()
        .clamp(0, 1)
        .cpu()
        .mul(255)
        .byte()
        .permute(1, 2, 0)
        .numpy()
    )


def match_targets_to_images(images: torch.Tensor, targets: torch.Tensor | None) -> torch.Tensor | None:
    if targets is None:
        return None
    cpu_targets = targets.detach().float().cpu().clamp(0, 1)
    if cpu_targets.shape[0] == 1 and images.shape[0] > 1:
        cpu_targets = cpu_targets.repeat(images.shape[0], 1, 1, 1)
    if cpu_targets.shape[0] != images.shape[0]:
        raise ValueError(
            f"Target batch size {cpu_targets.shape[0]} does not match images {images.shape[0]}."
        )
    if cpu_targets.shape[2:] != images.shape[2:]:
        cpu_targets = F.interpolate(
            cpu_targets,
            size=images.shape[2:],
            mode="bicubic",
            antialias=True,
        ).clamp(0, 1)
    return cpu_targets


@torch.no_grad()
def image_psnr(images: torch.Tensor, targets: torch.Tensor | None) -> torch.Tensor:
    cpu_images = images.detach().float().cpu().clamp(0, 1)
    cpu_targets = match_targets_to_images(cpu_images, targets)
    if cpu_targets is None:
        return torch.full((cpu_images.shape[0],), float("nan"))
    mse = (cpu_images - cpu_targets).pow(2).mean(dim=(1, 2, 3)).clamp_min(1.0e-10)
    return 10.0 * torch.log10(1.0 / mse)


@torch.no_grad()
def image_ssim(images: torch.Tensor, targets: torch.Tensor | None) -> torch.Tensor:
    cpu_images = images.detach().float().cpu().clamp(0, 1)
    cpu_targets = match_targets_to_images(cpu_images, targets)
    if cpu_targets is None:
        return torch.full((cpu_images.shape[0],), float("nan"))
    x = cpu_images.flatten(1)
    y = cpu_targets.flatten(1)
    mu_x = x.mean(dim=1)
    mu_y = y.mean(dim=1)
    var_x = (x - mu_x[:, None]).pow(2).mean(dim=1)
    var_y = (y - mu_y[:, None]).pow(2).mean(dim=1)
    cov_xy = ((x - mu_x[:, None]) * (y - mu_y[:, None])).mean(dim=1)
    c1 = 0.01**2
    c2 = 0.03**2
    ssim = ((2 * mu_x * mu_y + c1) * (2 * cov_xy + c2)) / (
        (mu_x.pow(2) + mu_y.pow(2) + c1) * (var_x + var_y + c2)
    )
    return ssim.clamp(-1.0, 1.0)


def side_by_side_image(left: torch.Tensor, right: torch.Tensor) -> np.ndarray:
    left_arr = tensor_to_uint8_hwc(left)
    right_arr = tensor_to_uint8_hwc(right)
    if left_arr.shape[:2] != right_arr.shape[:2]:
        raise ValueError(f"Image shapes differ: {left_arr.shape} vs {right_arr.shape}")
    divider = np.full((left_arr.shape[0], 4, 3), 255, dtype=np.uint8)
    return np.concatenate([left_arr, divider, right_arr], axis=1)


def selected_reward_from_detail(detail: Any, variant: str) -> float:
    if variant == "matched_mean_reward":
        return float(detail.matched_mean_reward)
    if variant == "final_reward":
        return float(detail.final_reward)
    if variant == "final_reward_norm":
        return float(detail.final_reward_norm)
    raise ValueError(f"Unsupported reward variant: {variant}")


def reward_table_row(
    detail: Any,
    selected_reward: float,
    psnr_value: float,
    ssim_value: float,
    cfg: Any,
) -> list[Any]:
    base_reward = selected_reward_from_detail(detail, str(cfg.reward.variant))
    return [
        detail.image_id,
        selected_reward,
        base_reward,
        selected_reward - base_reward,
        psnr_value,
        ssim_value,
        detail.num_gt,
        detail.num_pred,
        detail.num_matched,
        detail.missed,
        detail.false_positive,
        detail.matched_mean_reward,
        detail.final_reward,
        detail.final_reward_norm,
        detail.mean_iou_matched,
    ]


def reward_table_columns() -> list[str]:
    return [
        "image_id",
        "selected_reward",
        "base_reward",
        "psnr_bonus",
        "psnr",
        "ssim",
        "num_gt",
        "num_pred",
        "num_matched",
        "missed",
        "false_positive",
        "matched_mean_reward",
        "final_reward",
        "final_reward_norm",
        "mean_iou_matched",
    ]


def safe_filename(value: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in value)
    return safe[:160] or "image"


def save_tensor_png(image: torch.Tensor, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(tensor_to_uint8_hwc(image)).save(path)


def draw_reward_bar_graph(labels: list[str], values: list[float], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    width = max(760, 96 * max(len(values), 1))
    height = 520
    margin_left = 84
    margin_right = 32
    margin_top = 54
    margin_bottom = 94
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom

    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    draw.text((margin_left, 18), "Final selected reward by generated image", fill=(20, 20, 20))

    finite_values = [value for value in values if math.isfinite(value)]
    if not finite_values:
        draw.text((margin_left, margin_top), "No rewards", fill=(80, 80, 80))
        image.save(path)
        return

    vmin = min(0.0, min(finite_values))
    vmax = max(0.0, max(finite_values))
    if abs(vmax - vmin) < 1.0e-6:
        vmin -= 1.0
        vmax += 1.0
    pad = 0.08 * (vmax - vmin)
    vmin -= pad
    vmax += pad

    def y_for(value: float) -> float:
        return margin_top + (vmax - value) / (vmax - vmin) * plot_h

    x_axis = margin_left
    y_zero = y_for(0.0)
    draw.line((x_axis, margin_top, x_axis, margin_top + plot_h), fill=(80, 80, 80), width=1)
    draw.line((x_axis, y_zero, margin_left + plot_w, y_zero), fill=(120, 120, 120), width=1)
    draw.text((18, margin_top - 6), f"{vmax:.2f}", fill=(80, 80, 80))
    draw.text((18, margin_top + plot_h - 8), f"{vmin:.2f}", fill=(80, 80, 80))
    draw.text((28, y_zero - 8), "0", fill=(80, 80, 80))

    count = len(values)
    slot = plot_w / count
    bar_w = max(18, min(54, slot * 0.62))
    for index, (label, value) in enumerate(zip(labels, values)):
        plot_value = value if math.isfinite(value) else 0.0
        center = margin_left + slot * (index + 0.5)
        x0 = center - bar_w / 2
        x1 = center + bar_w / 2
        y_value = y_for(plot_value)
        y0 = min(y_zero, y_value)
        y1 = max(y_zero, y_value)
        color = (50, 111, 168) if plot_value >= 0 else (203, 82, 70)
        draw.rectangle((x0, y0, x1, y1), fill=color)
        value_y = y0 - 18 if plot_value >= 0 else y1 + 4
        value_text = f"{value:.3f}" if math.isfinite(value) else "nan"
        draw.text((x0 - 8, value_y), value_text, fill=(30, 30, 30))
        draw.text((x0 + 2, margin_top + plot_h + 18), label, fill=(30, 30, 30))

    image.save(path)


def save_training_step_artifacts(
    output_dir: Path,
    cfg: Any,
    step: int,
    images: torch.Tensor,
    targets: torch.Tensor | None,
    rewards: torch.Tensor,
    details: list[Any],
) -> None:
    every = int(cfg.reward.get("keep_images_every", 25))
    if every <= 0 or step % every != 0:
        return

    cpu_images = images.detach().float().cpu().clamp(0, 1)
    cpu_targets = match_targets_to_images(cpu_images, targets)
    psnr_values = image_psnr(cpu_images, cpu_targets)
    ssim_values = image_ssim(cpu_images, cpu_targets)
    reward_values = rewards.detach().float().cpu()

    step_dir = output_dir / "train_artifacts" / f"step_{step:07d}"
    image_dir = step_dir / "inference"
    image_dir.mkdir(parents=True, exist_ok=True)

    if cpu_targets is not None and cpu_targets.numel() > 0:
        save_tensor_png(cpu_targets[0], step_dir / "gt.png")

    rows: list[dict[str, Any]] = []
    bar_labels: list[str] = []
    bar_values: list[float] = []
    for idx, (image, detail, reward) in enumerate(zip(cpu_images, details, reward_values)):
        final_selected_reward = float(reward.item())
        base_reward = selected_reward_from_detail(detail, str(cfg.reward.variant))
        filename = f"{idx + 1:02d}_{safe_filename(detail.image_id)}.png"
        save_tensor_png(image, image_dir / filename)
        row = {
            "group_index": idx + 1,
            "image_id": detail.image_id,
            "generated_path": str(Path("inference") / filename),
            "final_selected_reward": final_selected_reward,
            "base_reward_without_psnr_bonus": float(base_reward),
            "psnr_bonus": float(final_selected_reward - base_reward),
            "psnr": float(psnr_values[idx].item()),
            "ssim": float(ssim_values[idx].item()),
            "num_gt": int(detail.num_gt),
            "num_pred": int(detail.num_pred),
            "num_matched": int(detail.num_matched),
            "missed": int(detail.missed),
            "false_positive": int(detail.false_positive),
            "matched_mean_reward": float(detail.matched_mean_reward),
            "final_reward": float(detail.final_reward),
            "final_reward_norm": float(detail.final_reward_norm),
            "mean_iou_matched": float(detail.mean_iou_matched),
        }
        rows.append(row)
        bar_labels.append(f"g{idx + 1}")
        bar_values.append(final_selected_reward)

    csv_path = step_dir / "rewards.csv"
    fieldnames = list(rows[0].keys()) if rows else [
        "group_index",
        "image_id",
        "generated_path",
        "final_selected_reward",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "step": step,
        "reward_variant": str(cfg.reward.variant),
        "num_generated": len(rows),
        "gt_path": "gt.png" if cpu_targets is not None and cpu_targets.numel() > 0 else None,
        "final_selected_reward_mean": (
            float(reward_values.mean().item()) if reward_values.numel() else float("nan")
        ),
        "final_selected_reward_min": (
            float(reward_values.min().item()) if reward_values.numel() else float("nan")
        ),
        "final_selected_reward_max": (
            float(reward_values.max().item()) if reward_values.numel() else float("nan")
        ),
        "rows": rows,
    }
    (step_dir / "rewards.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    draw_reward_bar_graph(bar_labels, bar_values, step_dir / "final_reward_bar.png")


def save_evaluation_step_artifacts(
    output_dir: Path,
    cfg: Any,
    step: int,
    images: torch.Tensor,
    targets: torch.Tensor | None,
    rewards: torch.Tensor,
    details: list[Any],
) -> None:
    cpu_images = images.detach().float().cpu().clamp(0, 1)
    cpu_targets = match_targets_to_images(cpu_images, targets)
    psnr_values = image_psnr(cpu_images, cpu_targets)
    ssim_values = image_ssim(cpu_images, cpu_targets)
    reward_values = rewards.detach().float().cpu()

    step_dir = output_dir / "eval_artifacts" / f"step_{step:07d}"
    image_dir = step_dir / "inference"
    gt_dir = step_dir / "gt"
    image_dir.mkdir(parents=True, exist_ok=True)
    if cpu_targets is not None:
        gt_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    bar_labels: list[str] = []
    bar_values: list[float] = []
    for idx, (image, detail, reward) in enumerate(zip(cpu_images, details, reward_values)):
        final_selected_reward = float(reward.item())
        base_reward = selected_reward_from_detail(detail, str(cfg.reward.variant))
        filename = f"{idx + 1:02d}_{safe_filename(detail.image_id)}.png"
        generated_path = Path("inference") / filename
        gt_path = Path("gt") / filename
        save_tensor_png(image, step_dir / generated_path)
        if cpu_targets is not None:
            save_tensor_png(cpu_targets[idx], step_dir / gt_path)
        row = {
            "eval_index": idx + 1,
            "image_id": detail.image_id,
            "generated_path": str(generated_path),
            "gt_path": str(gt_path) if cpu_targets is not None else "",
            "final_selected_reward": final_selected_reward,
            "base_reward_without_psnr_bonus": float(base_reward),
            "psnr_bonus": float(final_selected_reward - base_reward),
            "psnr": float(psnr_values[idx].item()),
            "ssim": float(ssim_values[idx].item()),
            "num_gt": int(detail.num_gt),
            "num_pred": int(detail.num_pred),
            "num_matched": int(detail.num_matched),
            "missed": int(detail.missed),
            "false_positive": int(detail.false_positive),
            "matched_mean_reward": float(detail.matched_mean_reward),
            "final_reward": float(detail.final_reward),
            "final_reward_norm": float(detail.final_reward_norm),
            "mean_iou_matched": float(detail.mean_iou_matched),
        }
        rows.append(row)
        bar_labels.append(f"e{idx + 1}")
        bar_values.append(final_selected_reward)

    csv_path = step_dir / "rewards.csv"
    fieldnames = list(rows[0].keys()) if rows else [
        "eval_index",
        "image_id",
        "generated_path",
        "gt_path",
        "final_selected_reward",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "step": step,
        "reward_variant": str(cfg.reward.variant),
        "num_eval_images": len(rows),
        "final_selected_reward_mean": (
            float(reward_values.mean().item()) if reward_values.numel() else float("nan")
        ),
        "final_selected_reward_min": (
            float(reward_values.min().item()) if reward_values.numel() else float("nan")
        ),
        "final_selected_reward_max": (
            float(reward_values.max().item()) if reward_values.numel() else float("nan")
        ),
        "rows": rows,
    }
    (step_dir / "rewards.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    draw_reward_bar_graph(bar_labels, bar_values, step_dir / "final_reward_bar.png")


def autocast_context(device: torch.device, precision: str):
    if device.type != "cuda" or precision == "fp32":
        return nullcontext()
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}[precision]
    return torch.autocast(device_type=device.type, dtype=dtype)


def load_diffbir(cfg: Any, device: torch.device) -> tuple[ControlLDM, SwinIR, Diffusion]:
    model_cfg = OmegaConf.load(cfg.paths.diffbir_train_config)

    cldm: ControlLDM = instantiate_from_config(model_cfg.model.cldm)
    sd = torch.load(cfg.paths.sd_path, map_location="cpu")["state_dict"]
    unused, missing = cldm.load_pretrained_sd(sd)
    print(f"Loaded SD: unused={len(unused)} missing={len(missing)}")

    control_weight = torch.load(cfg.paths.controlnet_path, map_location="cpu")
    cldm.load_controlnet_from_ckpt(control_weight)
    print(f"Loaded DiffBIR controlnet: {cfg.paths.controlnet_path}")

    if cfg.train.precision != "fp32":
        dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}[cfg.train.precision]
        cldm.cast_dtype(dtype)

    swinir: SwinIR = instantiate_from_config(model_cfg.model.swinir)
    swinir_weight = torch.load(cfg.paths.swinir_path, map_location="cpu")
    if "state_dict" in swinir_weight:
        swinir_weight = swinir_weight["state_dict"]
    swinir_weight = {
        (k[len("module.") :] if k.startswith("module.") else k): v
        for k, v in swinir_weight.items()
    }
    swinir.load_state_dict(swinir_weight, strict=True)
    swinir.eval().to(device)
    for p in swinir.parameters():
        p.requires_grad = False

    diffusion: Diffusion = instantiate_from_config(model_cfg.model.diffusion)
    diffusion.to(device)

    cldm.to(device)
    cldm.unet.eval()
    cldm.vae.eval()
    cldm.clip.eval()
    cldm.controlnet.train()
    return cldm, swinir, diffusion


def apply_lora(cldm: ControlLDM, cfg: Any) -> None:
    reports = []
    roots = [str(cfg.lora.target_root)]
    if roots == ["all"]:
        roots = ["controlnet", "unet"]
    for root in roots:
        module = getattr(cldm, root)
        report = inject_lora(
            module,
            rank=int(cfg.lora.rank),
            alpha=float(cfg.lora.alpha),
            dropout=float(cfg.lora.dropout),
            target_keywords=list(cfg.lora.target_keywords),
            include_linear=bool(cfg.lora.include_linear),
            include_conv1x1=bool(cfg.lora.include_conv1x1),
        )
        reports.append((root, report))
    for root, report in reports:
        print(
            f"LoRA root={root}: wrapped={report.wrapped} "
            f"trainable={report.trainable_params:,}/{report.total_params:,}"
        )


def build_reward(cfg: Any):
    if cfg.reward.backend == "bridge":
        return BridgeReward(
            config_path=cfg.reward.config,
            pipeline_root=cfg.reward.pipeline_root,
            work_dir=cfg.reward.work_dir,
            iou_threshold=float(cfg.reward.iou_threshold),
            bridge_score_threshold=float(cfg.reward.bridge_score_threshold),
            miss_penalty=float(cfg.reward.miss_penalty),
            false_positive_penalty=float(cfg.reward.false_positive_penalty),
            variant=str(cfg.reward.variant),
            keep_images_every=int(cfg.reward.keep_images_every),
            bridge_env_python=str(cfg.reward.bridge_env_python),
        )
    if cfg.reward.backend == "constant":
        return ConstantReward(float(cfg.reward.constant_value), str(cfg.reward.variant))
    raise ValueError(f"Unsupported reward backend: {cfg.reward.backend}")


def algorithm_name(cfg: Any) -> str:
    return str(cfg.train.get("algorithm", "ddpo")).lower()


def algorithm_cfg(cfg: Any) -> Any:
    algo = algorithm_name(cfg)
    if algo == "ddpo":
        return cfg.ddpo
    if algo == "grpo":
        return cfg.grpo
    raise ValueError(f"Unsupported train.algorithm: {algo}")


@torch.no_grad()
def apply_cleaner(swinir: SwinIR, lq: torch.Tensor) -> torch.Tensor:
    h0, w0 = lq.shape[2:]
    lq_pad = pad_to_multiples_of(lq, multiple=64)
    return swinir(lq_pad)[:, :, :h0, :w0].clamp(0, 1)


def prepare_conditions(
    cldm: ControlLDM,
    diffusion: Diffusion,
    clean: torch.Tensor,
    prompts: list[str],
    negative_prompts: list[str],
    noise_aug: int,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    cond = cldm.prepare_condition(clean, prompts)
    uncond = cldm.prepare_condition(clean, negative_prompts)
    cond["c_img"] = pad_to_multiples_of(cond["c_img"], multiple=8)
    uncond["c_img"] = pad_to_multiples_of(uncond["c_img"], multiple=8)
    if noise_aug > 0:
        t = torch.full(
            (clean.shape[0],),
            fill_value=noise_aug,
            device=clean.device,
            dtype=torch.long,
        )
        cond["c_img"] = diffusion.q_sample(cond["c_img"], t=t, noise=torch.randn_like(cond["c_img"]))
        uncond["c_img"] = cond["c_img"].detach().clone()
    return cond, uncond


@torch.no_grad()
def rollout(
    cldm: ControlLDM,
    swinir: SwinIR,
    diffusion: Diffusion,
    batch: dict[str, Any],
    sampler: DDPOSpacedSampler,
    cfg: Any,
    device: torch.device,
    logprob_reduce: str,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, torch.Tensor], torch.Tensor]:
    lq = batch["lq"].to(device, non_blocking=True)
    clean = apply_cleaner(swinir, lq)
    cond, uncond = prepare_conditions(
        cldm,
        diffusion,
        clean,
        batch["prompts"],
        batch["negative_prompts"],
        int(cfg.sample.noise_aug),
    )
    x_T = None
    if cfg.sample.start_point_type == "cond":
        x_T = diffusion.q_sample(
            cond["c_img"],
            t=torch.full(
                (lq.shape[0],),
                diffusion.num_timesteps - 1,
                dtype=torch.long,
                device=device,
            ),
            noise=torch.randn_like(cond["c_img"]),
        )

    z, trajectory = sampler.sample_with_log_probs(
        model=cldm,
        device=device,
        steps=int(cfg.sample.steps),
        x_size=tuple(cond["c_img"].shape),
        cond=cond,
        uncond=uncond,
        cfg_scale=float(cfg.sample.cfg_scale),
        x_T=x_T,
        logprob_reduce=logprob_reduce,
        progress=bool(cfg.train.get("sample_progress", False)),
    )
    decoded = cldm.vae_decode(z)[:, :, : clean.shape[2], : clean.shape[3]]
    images = wavelet_reconstruction((decoded + 1) / 2, clean).clamp(0, 1)
    images = F.interpolate(
        images,
        size=lq.shape[2:],
        mode="bicubic",
        antialias=True,
    ).clamp(0, 1)
    return images, trajectory, cond, uncond, clean


def make_grpo_microbatch(batch: dict[str, Any], start: int, count: int) -> dict[str, Any]:
    if len(batch["image_ids"]) != 1:
        raise ValueError("GRPO mode expects train.batch_size=1: one source image per group.")
    base_id = batch["image_ids"][0]
    indices = range(start, start + count)
    micro = {
        "image_ids": [f"{base_id}_g{i:02d}" for i in indices],
        "lq": batch["lq"][0:1].repeat(count, 1, 1, 1),
        "gt_instances": [batch["gt_instances"][0] for _ in indices],
        "prompts": [batch["prompts"][0] for _ in indices],
        "negative_prompts": [batch["negative_prompts"][0] for _ in indices],
    }
    if "hq" in batch:
        micro["hq"] = batch["hq"][0:1].repeat(count, 1, 1, 1)
    return micro


def cat_trajectory(chunks: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    return {key: torch.cat([chunk[key] for chunk in chunks], dim=0) for key in chunks[0]}


def cat_condition(chunks: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    return {key: torch.cat([chunk[key].detach() for chunk in chunks], dim=0) for key in chunks[0]}


@torch.no_grad()
def rollout_grpo_group(
    cldm: ControlLDM,
    swinir: SwinIR,
    diffusion: Diffusion,
    batch: dict[str, Any],
    sampler: DDPOSpacedSampler,
    cfg: Any,
    device: torch.device,
) -> tuple[
    torch.Tensor,
    dict[str, torch.Tensor],
    dict[str, torch.Tensor],
    dict[str, torch.Tensor],
    list[str],
    list[Any],
]:
    group_size = int(cfg.grpo.group_size)
    microbatch = int(cfg.grpo.generation_microbatch)
    if group_size <= 1:
        raise ValueError("GRPO group_size must be >= 2.")
    if microbatch <= 0:
        raise ValueError("GRPO generation_microbatch must be positive.")

    images: list[torch.Tensor] = []
    trajectories: list[dict[str, torch.Tensor]] = []
    conds: list[dict[str, torch.Tensor]] = []
    unconds: list[dict[str, torch.Tensor]] = []
    image_ids: list[str] = []
    gt_instances: list[Any] = []

    for start in range(0, group_size, microbatch):
        count = min(microbatch, group_size - start)
        micro = make_grpo_microbatch(batch, start, count)
        micro_images, micro_traj, micro_cond, micro_uncond, _ = rollout(
            cldm,
            swinir,
            diffusion,
            micro,
            sampler,
            cfg,
            device,
            logprob_reduce=str(cfg.grpo.logprob_reduce),
        )
        images.append(micro_images.detach().cpu())
        trajectories.append(micro_traj)
        conds.append(micro_cond)
        unconds.append(micro_uncond)
        image_ids.extend(micro["image_ids"])
        gt_instances.extend(micro["gt_instances"])

    return (
        torch.cat(images, dim=0),
        cat_trajectory(trajectories),
        cat_condition(conds),
        cat_condition(unconds),
        image_ids,
        gt_instances,
    )


def compute_advantages(
    rewards: torch.Tensor,
    baseline: float,
    cfg: Any,
) -> tuple[torch.Tensor, float]:
    clipped = rewards.clamp(float(cfg.reward.clip_min), float(cfg.reward.clip_max))
    if cfg.ddpo.advantage_mode == "raw":
        adv = clipped
        new_baseline = baseline
    elif cfg.ddpo.advantage_mode == "ema":
        adv = clipped - baseline
        mean_reward = float(clipped.mean().item())
        new_baseline = float(cfg.ddpo.baseline_beta) * baseline + (
            1.0 - float(cfg.ddpo.baseline_beta)
        ) * mean_reward
    elif cfg.ddpo.advantage_mode == "batch_norm":
        adv = clipped - clipped.mean()
        new_baseline = baseline
    else:
        raise ValueError(f"Unsupported advantage mode: {cfg.ddpo.advantage_mode}")
    if bool(cfg.ddpo.normalize_advantage) and adv.numel() > 1:
        adv = adv / adv.std(unbiased=False).clamp_min(1e-6)
    return adv * float(cfg.ddpo.reward_scale), new_baseline


def compute_grpo_advantages(
    rewards: torch.Tensor,
    cfg: Any,
) -> tuple[torch.Tensor, dict[str, float]]:
    clipped = rewards.clamp(float(cfg.reward.clip_min), float(cfg.reward.clip_max))
    mean = clipped.mean()
    std = clipped.std(unbiased=False)
    advantages = clipped - mean
    if bool(cfg.grpo.normalize_advantage):
        advantages = advantages / (std + float(cfg.grpo.advantage_epsilon))
    advantages = torch.nan_to_num(advantages) * float(cfg.grpo.reward_scale)
    return advantages, {
        "group_reward_mean": float(mean.item()),
        "group_reward_std": float(std.item()),
    }


def ppo_policy_loss(
    new_log_probs: torch.Tensor,
    reference_kls: torch.Tensor,
    trajectory: dict[str, torch.Tensor],
    advantages: torch.Tensor,
    clip_range: float,
    kl_coef: float,
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, float]]:
    old_log_probs = trajectory["old_log_probs"].to(device)
    mask = trajectory["mask"].to(device)
    adv = advantages.to(device).view(-1, 1)
    log_ratio = (new_log_probs - old_log_probs).clamp(-20, 20)
    ratio = torch.exp(log_ratio)
    clipped_ratio = ratio.clamp(1.0 - clip_range, 1.0 + clip_range)
    unclipped = ratio * adv
    clipped = clipped_ratio * adv
    denom = mask.sum().clamp_min(1.0)
    policy_loss = -(torch.minimum(unclipped, clipped) * mask).sum() / denom
    approx_kl = (0.5 * log_ratio.pow(2) * mask).sum() / denom
    reference_kl = (reference_kls.to(device) * mask).sum() / denom
    kl_loss = kl_coef * reference_kl
    loss = policy_loss + kl_loss
    clip_frac = (((ratio - 1.0).abs() > clip_range).float() * mask).sum() / denom
    stats = {
        "policy_loss": float(policy_loss.detach().cpu().item()),
        "grpo_loss": float(policy_loss.detach().cpu().item()),
        "kl_loss": float(kl_loss.detach().cpu().item()),
        "total_loss": float(loss.detach().cpu().item()),
        "approx_kl": float(approx_kl.detach().cpu().item()),
        "reference_kl": float(reference_kl.detach().cpu().item()),
        "clip_frac": float(clip_frac.detach().cpu().item()),
        "ratio_mean": float(((ratio * mask).sum() / denom).detach().cpu().item()),
    }
    return loss, stats


def optimize_policy(
    cldm: ControlLDM,
    sampler: DDPOSpacedSampler,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    trainable_params: list[torch.nn.Parameter],
    trajectory: dict[str, torch.Tensor],
    cond: dict[str, torch.Tensor],
    uncond: dict[str, torch.Tensor],
    advantages: torch.Tensor,
    cfg: Any,
    algo_cfg: Any,
    device: torch.device,
) -> dict[str, float]:
    last_stats: dict[str, float] = {}
    for _ in range(int(algo_cfg.ppo_epochs)):
        optimizer.zero_grad(set_to_none=True)
        with autocast_context(device, str(cfg.train.precision)):
            new_log_probs = sampler.log_probs_for_trajectory(
                cldm,
                trajectory,
                cond,
                uncond,
                device,
                str(algo_cfg.logprob_reduce),
            )
            reference_kls = sampler.reference_kls_for_trajectory(
                cldm,
                trajectory,
                cond,
                uncond,
                device,
                str(algo_cfg.logprob_reduce),
                reference_context=lambda: temporary_lora_enabled(cldm, False),
            )
            loss, loss_stats = ppo_policy_loss(
                new_log_probs,
                reference_kls,
                trajectory,
                advantages,
                clip_range=float(algo_cfg.clip_range),
                kl_coef=float(algo_cfg.kl_coef),
                device=device,
            )
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(
            trainable_params,
            max_norm=float(cfg.train.max_grad_norm),
        )
        scaler.step(optimizer)
        scaler.update()
        last_stats = loss_stats
        last_stats["grad_norm"] = float(
            grad_norm.detach().cpu().item() if torch.is_tensor(grad_norm) else grad_norm
        )
        last_stats["lora_param_l2_norm"] = parameter_l2_norm(trainable_params)
        last_stats["loss"] = float(loss.detach().cpu().item())
    return last_stats


def capture_rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: Any) -> None:
    if not isinstance(state, dict):
        return
    if "python" in state:
        random.setstate(state["python"])
    if "numpy" in state:
        np.random.set_state(state["numpy"])
    if "torch" in state:
        torch.set_rng_state(state["torch"])
    cuda_states = state.get("cuda")
    if cuda_states is not None and torch.cuda.is_available():
        for device_index, cuda_state in enumerate(cuda_states[: torch.cuda.device_count()]):
            torch.cuda.set_rng_state(cuda_state, device_index)


def save_checkpoint(
    path: Path,
    cldm: ControlLDM,
    optimizer: torch.optim.Optimizer,
    step: int,
    cfg: Any,
    scaler: Any | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "step": step,
        "lora": lora_state_dict(cldm),
        "optimizer": optimizer.state_dict(),
        "config": OmegaConf.to_container(cfg, resolve=True),
        "rng_state": capture_rng_state(),
    }
    if scaler is not None:
        payload["scaler"] = scaler.state_dict()
    torch.save(payload, path)


def checkpoint_step_from_name(path: Path) -> int | None:
    name = path.name
    prefix = "lora_step_"
    suffix = ".pt"
    if not (name.startswith(prefix) and name.endswith(suffix)):
        return None
    try:
        return int(name[len(prefix) : -len(suffix)])
    except ValueError:
        return None


def find_latest_checkpoint(checkpoint_dir: Path) -> Path | None:
    if not checkpoint_dir.is_dir():
        return None
    candidates: list[tuple[int, int, float, str, Path]] = []
    for path in checkpoint_dir.glob("lora_step_*.pt"):
        step = checkpoint_step_from_name(path)
        if step is not None:
            candidates.append((0, step, path.stat().st_mtime, str(path), path))
    final_path = checkpoint_dir / "lora_final.pt"
    if final_path.is_file():
        candidates.append((1, 0, final_path.stat().st_mtime, str(final_path), final_path))
    if not candidates:
        return None
    return max(candidates)[4]


def resolve_resume_checkpoint(value: Any, output_dir: Path) -> Path | None:
    value = maybe_null(value)
    if value is None:
        return None
    token = str(value).strip()
    if token.lower() in {"0", "false", "no", "off"}:
        return None
    if token.lower() in {"1", "true", "yes", "latest", "auto"}:
        checkpoint = find_latest_checkpoint(output_dir / "checkpoints")
        if checkpoint is None:
            print(f"No checkpoint found in {output_dir / 'checkpoints'}; starting from scratch.")
        return checkpoint
    checkpoint = Path(os.path.expandvars(token)).expanduser()
    if not checkpoint.is_absolute():
        checkpoint = output_dir / checkpoint
    return checkpoint


def load_checkpoint(
    path: Path,
    cldm: ControlLDM,
    optimizer: torch.optim.Optimizer,
    scaler: Any | None,
    device: torch.device,
) -> int:
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location=device)

    lora_weights = checkpoint.get("lora")
    if not isinstance(lora_weights, dict):
        raise ValueError(f"Checkpoint has no LoRA weights: {path}")

    model_state = cldm.state_dict()
    expected_lora_keys = set(lora_state_dict(cldm).keys())
    checkpoint_keys = set(lora_weights.keys())
    missing = sorted(expected_lora_keys - checkpoint_keys)
    unexpected = sorted(checkpoint_keys - expected_lora_keys)
    if missing or unexpected:
        raise RuntimeError(
            f"Checkpoint LoRA keys do not match current model. "
            f"missing={missing[:5]} unexpected={unexpected[:5]}"
        )
    for key, value in lora_weights.items():
        current = model_state[key]
        if tuple(current.shape) != tuple(value.shape):
            raise RuntimeError(
                f"Checkpoint tensor shape mismatch for {key}: "
                f"checkpoint={tuple(value.shape)} current={tuple(current.shape)}"
            )
        current.copy_(value.to(device=current.device, dtype=current.dtype))

    optimizer.load_state_dict(checkpoint["optimizer"])
    if scaler is not None and "scaler" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler"])
    restore_rng_state(checkpoint.get("rng_state"))
    step = int(checkpoint.get("step", 0))
    print(f"Resumed checkpoint: {path} at step {step}")
    return step


def rotate_dataset_for_resume(dataset: SATextParquetDataset, consumed_samples: int) -> int:
    if consumed_samples <= 0 or len(dataset) == 0:
        return 0
    offset = consumed_samples % len(dataset)
    if offset == 0:
        return 0
    dataset.rows = dataset.rows[offset:] + dataset.rows[:offset]
    return offset


def build_dataset_from_section(section: Any, fallback: Any | None = None) -> SATextParquetDataset:
    fallback = fallback or {}
    return SATextParquetDataset(
        parquet_path=section.get("parquet", fallback.get("parquet", None)),
        manifest_path=section.get("manifest", fallback.get("manifest", None)),
        level=int(section.get("level", fallback.get("level", 2))),
        start=int(section.get("start", fallback.get("start", 0))),
        max_samples=(
            None
            if section.get("max_samples", fallback.get("max_samples", None)) is None
            else int(section.get("max_samples", fallback.get("max_samples", None)))
        ),
        image_size=int(section.get("image_size", fallback.get("image_size", 512))),
        prompt_mode=str(section.get("prompt_mode", fallback.get("prompt_mode", "gt_text"))),
        fixed_prompt=str(
            section.get(
                "fixed_prompt",
                fallback.get("fixed_prompt", "A high-quality restored image with clear readable text."),
            )
        ),
        negative_prompt=str(section.get("negative_prompt", fallback.get("negative_prompt", ""))),
    )


def build_fixed_eval_batch(cfg: Any) -> tuple[dict[str, Any], str] | None:
    eval_cfg = cfg.get("eval", {})
    if not bool(eval_cfg.get("enabled", False)):
        return None
    num_images = int(eval_cfg.get("num_images", 5))
    eval_section = OmegaConf.merge(
        cfg.data,
        OmegaConf.create(
            {
                "parquet": eval_cfg.get("parquet", cfg.data.get("parquet", None)),
                "manifest": eval_cfg.get("manifest", None),
                "level": eval_cfg.get("level", cfg.data.get("level", 2)),
                "start": eval_cfg.get("start", 0),
                "max_samples": eval_cfg.get("max_samples", num_images),
                "image_size": eval_cfg.get("image_size", cfg.data.get("image_size", 512)),
                "prompt_mode": eval_cfg.get("prompt_mode", cfg.data.get("prompt_mode", "gt_text")),
                "fixed_prompt": eval_cfg.get("fixed_prompt", cfg.data.get("fixed_prompt", "")),
                "negative_prompt": eval_cfg.get("negative_prompt", cfg.data.get("negative_prompt", "")),
            }
        ),
    )
    dataset = build_dataset_from_section(eval_section, cfg.data)
    num_images = min(num_images, len(dataset))
    if num_images <= 0:
        return None
    batch = collate_sa_text([dataset[index] for index in range(num_images)])
    return batch, dataset.source_description


def slice_collated_batch(batch: dict[str, Any], start: int, end: int) -> dict[str, Any]:
    out = {
        "image_ids": batch["image_ids"][start:end],
        "lq": batch["lq"][start:end],
        "hq": batch["hq"][start:end],
        "gt_instances": batch["gt_instances"][start:end],
        "prompts": batch["prompts"][start:end],
        "negative_prompts": batch["negative_prompts"][start:end],
    }
    return out


def log_wandb_evaluation(
    wandb_run: Any,
    cfg: Any,
    step: int,
    images: torch.Tensor,
    targets: torch.Tensor | None,
    rewards: torch.Tensor,
    details: list[Any],
    eval_state: dict[str, float],
) -> None:
    if wandb_run is None:
        return

    cpu_images = images.detach().float().cpu().clamp(0, 1)
    cpu_targets = match_targets_to_images(cpu_images, targets)
    psnr_values = image_psnr(cpu_images, cpu_targets)
    ssim_values = image_ssim(cpu_images, cpu_targets)
    reward_values = rewards.detach().float().cpu()

    eval_state["reward_total"] = float(eval_state.get("reward_total", 0.0)) + float(
        reward_values.sum().item()
    )
    eval_state["reward_count"] = float(eval_state.get("reward_count", 0.0)) + float(
        reward_values.numel()
    )
    cumulative_mean = eval_state["reward_total"] / max(eval_state["reward_count"], 1.0)

    log_visuals = bool(cfg.get("wandb", {}).get("log_visuals", True))
    if log_visuals:
        import wandb

        table = wandb.Table(columns=["output_gt", *reward_table_columns()])
    else:
        table = None
    payload: dict[str, Any] = {
        "train/step": step,
        "eval/reward_mean": float(reward_values.mean().item()),
        "eval/reward_cumulative_mean": float(cumulative_mean),
        "eval/psnr_mean": float(psnr_values.mean().item()),
        "eval/ssim_mean": float(ssim_values.mean().item()),
    }
    for idx, (image, detail, reward) in enumerate(zip(cpu_images, details, reward_values), start=1):
        gt_image = cpu_targets[idx - 1] if cpu_targets is not None else image
        psnr_value = float(psnr_values[idx - 1].item())
        ssim_value = float(ssim_values[idx - 1].item())
        selected_reward = float(reward.item())
        key = f"eval/{step}_{idx}"
        side_by_side = side_by_side_image(image, gt_image)
        payload[f"{key}_reward"] = selected_reward
        payload[f"{key}_psnr"] = psnr_value
        payload[f"{key}_ssim"] = ssim_value
        payload[f"{key}_matched_mean_reward"] = float(detail.matched_mean_reward)
        payload[f"{key}_final_reward"] = float(detail.final_reward)
        payload[f"{key}_final_reward_norm"] = float(detail.final_reward_norm)
        if log_visuals and table is not None:
            payload[key] = wandb.Image(
                side_by_side,
                caption=(
                    f"{detail.image_id} | output left, GT right | "
                    f"reward={selected_reward:.4f} psnr={psnr_value:.2f} ssim={ssim_value:.4f}"
                ),
            )
            table.add_data(
                wandb.Image(side_by_side, caption=f"{detail.image_id} output | GT"),
                *reward_table_row(detail, selected_reward, psnr_value, ssim_value, cfg),
            )
    if log_visuals and table is not None:
        payload["eval/reward_table"] = table
    wandb_run.log(payload)


@torch.no_grad()
def run_fixed_evaluation(
    cldm: ControlLDM,
    swinir: SwinIR,
    diffusion: Diffusion,
    sampler: DDPOSpacedSampler,
    reward_fn: Any,
    eval_batch: dict[str, Any] | None,
    cfg: Any,
    algo_cfg: Any,
    device: torch.device,
    step: int,
    wandb_run: Any,
    eval_state: dict[str, float],
    eval_metrics_path: Path,
    output_dir: Path,
) -> None:
    eval_cfg = cfg.get("eval", {})
    every = int(eval_cfg.get("every", 0))
    if eval_batch is None or every <= 0 or step % every != 0:
        return

    saved_rng = capture_rng_state()
    was_training = cldm.controlnet.training
    try:
        release_cuda_cache(device)
        seed_everything(int(eval_cfg.get("seed", int(cfg.train.seed) + 100_000)))
        cldm.controlnet.eval()
        batch_size = max(1, int(eval_cfg.get("batch_size", len(eval_batch["image_ids"]))))
        all_images: list[torch.Tensor] = []
        all_rewards: list[torch.Tensor] = []
        all_details: list[Any] = []
        all_targets: list[torch.Tensor] = []
        for chunk_index, start in enumerate(range(0, len(eval_batch["image_ids"]), batch_size)):
            end = min(start + batch_size, len(eval_batch["image_ids"]))
            chunk = slice_collated_batch(eval_batch, start, end)
            images, trajectory, cond, uncond, clean = rollout(
                cldm,
                swinir,
                diffusion,
                chunk,
                sampler,
                cfg,
                device,
                logprob_reduce=str(algo_cfg.logprob_reduce),
            )
            images_cpu = images.detach().cpu()
            targets_cpu = chunk["hq"].detach().cpu() if "hq" in chunk else None
            del images, trajectory, cond, uncond, clean
            release_cuda_cache(device)

            reward_step = step * 1000 + 999 + chunk_index
            rewards_cpu, details = reward_fn(
                chunk["image_ids"],
                images_cpu,
                chunk["gt_instances"],
                reward_step,
            )
            rewards_cpu, _ = add_psnr_reward_bonus(
                rewards_cpu,
                images_cpu,
                targets_cpu,
                cfg,
            )
            all_images.append(images_cpu)
            all_rewards.append(rewards_cpu.detach().cpu())
            all_details.extend(details)
            if targets_cpu is not None:
                all_targets.append(targets_cpu)
            release_cuda_cache(device)
        images_cpu = torch.cat(all_images, dim=0)
        rewards_cpu = torch.cat(all_rewards, dim=0)
        targets_cpu = torch.cat(all_targets, dim=0) if all_targets else None
        psnr_values = image_psnr(images_cpu, targets_cpu)
        ssim_values = image_ssim(images_cpu, targets_cpu)
        with eval_metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {
                        "step": step,
                        "reward_mean": float(rewards_cpu.mean().item()),
                        "psnr_mean": float(psnr_values.mean().item()),
                        "ssim_mean": float(ssim_values.mean().item()),
                        "image_ids": [detail.image_id for detail in all_details],
                        "rewards": [float(v) for v in rewards_cpu.tolist()],
                        "psnr": [float(v) for v in psnr_values.tolist()],
                        "ssim": [float(v) for v in ssim_values.tolist()],
                        "details": [asdict(detail) for detail in all_details],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
        log_wandb_evaluation(
            wandb_run,
            cfg,
            step,
            images_cpu,
            targets_cpu,
            rewards_cpu,
            all_details,
            eval_state,
        )
        save_evaluation_step_artifacts(
            output_dir,
            cfg,
            step,
            images_cpu,
            targets_cpu,
            rewards_cpu,
            all_details,
        )
    finally:
        if was_training:
            cldm.controlnet.train()
        release_cuda_cache(device)
        restore_rng_state(saved_rng)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config, args.overrides)
    seed_everything(int(cfg.train.seed))
    torch.backends.cuda.matmul.allow_tf32 = True
    algo = algorithm_name(cfg)
    algo_cfg = algorithm_cfg(cfg)
    if algo == "grpo" and int(cfg.train.batch_size) != 1:
        raise ValueError("GRPO currently expects train.batch_size=1; use grpo.group_size for samples per image.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("DDPO-LoRA fine-tuning is expected to run on a CUDA GPU.")

    output_dir = Path(cfg.train.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "resolved_config.yaml").write_text(OmegaConf.to_yaml(cfg), encoding="utf-8")
    metrics_path = output_dir / "metrics.jsonl"
    eval_metrics_path = output_dir / "eval_metrics.jsonl"
    wandb_run = init_wandb(cfg, output_dir, algo)

    dataset = build_dataset_from_section(cfg.data)
    print(f"Dataset: {len(dataset):,} samples from {dataset.source_description}")
    eval_result = build_fixed_eval_batch(cfg)
    if eval_result is None:
        eval_batch = None
    else:
        eval_batch, eval_source = eval_result
        print(f"Evaluation: {len(eval_batch['image_ids'])} fixed samples from {eval_source}")

    cldm, swinir, diffusion = load_diffbir(cfg, device)
    apply_lora(cldm, cfg)
    trainable_params = [p for p in cldm.parameters() if p.requires_grad]
    if not trainable_params:
        raise RuntimeError("No trainable LoRA parameters were created.")
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=float(cfg.train.learning_rate),
        betas=(float(cfg.train.adam_beta1), float(cfg.train.adam_beta2)),
        weight_decay=float(cfg.train.weight_decay),
    )
    scaler = torch.cuda.amp.GradScaler(enabled=(cfg.train.precision == "fp16"))
    resume_checkpoint = resolve_resume_checkpoint(
        cfg.train.get("resume_from_checkpoint", None),
        output_dir,
    )
    global_step = 0
    if resume_checkpoint is not None:
        global_step = load_checkpoint(resume_checkpoint, cldm, optimizer, scaler, device)
    shuffle_data = bool(cfg.train.get("shuffle_data", True))
    if not shuffle_data:
        consumed_samples = global_step * int(cfg.train.batch_size)
        dataset_offset = rotate_dataset_for_resume(dataset, consumed_samples)
        if global_step > 0:
            print(
                f"Sequential data resume: skipped {consumed_samples} consumed samples "
                f"(dataset offset {dataset_offset})."
            )
    elif global_step > 0:
        print("Resuming with train.shuffle_data=true; dataset order is randomized.")
    loader = DataLoader(
        dataset,
        batch_size=int(cfg.train.batch_size),
        shuffle=shuffle_data,
        num_workers=int(cfg.train.num_workers),
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_sa_text,
    )
    reward_fn = build_reward(cfg)
    sampler = DDPOSpacedSampler(
        diffusion.betas,
        diffusion.parameterization,
        rescale_cfg=bool(cfg.sample.rescale_cfg),
    )
    baseline = 0.0
    eval_state: dict[str, float] = {}
    train_steps = int(cfg.train.train_steps)
    progress_bar = make_progress_bar(
        enabled=bool(cfg.train.get("progress", False)),
        total=train_steps,
        desc=f"{algo.upper()} inference 0/{train_steps}",
        initial=global_step,
    )
    print(f"Training algorithm: {algo}")

    try:
        while global_step < train_steps:
            for batch in loader:
                global_step += 1
                source_image_id = batch["image_ids"][0] if batch["image_ids"] else "unknown"
                if progress_bar is not None:
                    progress_bar.set_description(f"{algo.upper()} inference {global_step}/{train_steps}")
                    progress_bar.set_postfix_str(f"image={source_image_id}", refresh=False)
                step_start = time.perf_counter()
                cldm.controlnet.train()
                cldm.control_scales = [float(cfg.sample.strength)] * 13

                rollout_start = time.perf_counter()
                if algo == "ddpo":
                    with autocast_context(device, str(cfg.train.precision)):
                        images, trajectory, cond, uncond, _ = rollout(
                            cldm,
                            swinir,
                            diffusion,
                            batch,
                            sampler,
                            cfg,
                            device,
                            logprob_reduce=str(cfg.ddpo.logprob_reduce),
                        )

                    rollout_seconds = time.perf_counter() - rollout_start
                    reward_start = time.perf_counter()
                    rewards_cpu, details = reward_fn(
                        batch["image_ids"],
                        images.detach().cpu(),
                        batch["gt_instances"],
                        global_step,
                    )
                    rewards_cpu, psnr_metrics = add_psnr_reward_bonus(
                        rewards_cpu,
                        images,
                        batch.get("hq"),
                        cfg,
                    )
                    reward_seconds = time.perf_counter() - reward_start
                    advantages, baseline = compute_advantages(rewards_cpu, baseline, cfg)
                    extra_metric = {
                        "baseline": baseline,
                        **psnr_metrics,
                    }
                elif algo == "grpo":
                    with autocast_context(device, str(cfg.train.precision)):
                        images, trajectory, cond, uncond, reward_image_ids, reward_gt = rollout_grpo_group(
                            cldm,
                            swinir,
                            diffusion,
                            batch,
                            sampler,
                            cfg,
                            device,
                        )
                    rollout_seconds = time.perf_counter() - rollout_start
                    reward_start = time.perf_counter()
                    rewards_cpu, details = reward_fn(
                        reward_image_ids,
                        images,
                        reward_gt,
                        global_step,
                    )
                    rewards_cpu, psnr_metrics = add_psnr_reward_bonus(
                        rewards_cpu,
                        images,
                        batch.get("hq"),
                        cfg,
                    )
                    reward_seconds = time.perf_counter() - reward_start
                    advantages, grpo_stats = compute_grpo_advantages(rewards_cpu, cfg)
                    extra_metric = {
                        **grpo_stats,
                        "group_size": int(cfg.grpo.group_size),
                        **psnr_metrics,
                    }
                else:
                    raise ValueError(f"Unsupported train.algorithm: {algo}")

                update_start = time.perf_counter()
                last_stats = optimize_policy(
                    cldm,
                    sampler,
                    optimizer,
                    scaler,
                    trainable_params,
                    trajectory,
                    cond,
                    uncond,
                    advantages,
                    cfg,
                    algo_cfg,
                    device,
                )
                update_seconds = time.perf_counter() - update_start
                step_seconds = time.perf_counter() - step_start

                reward_mean = float(rewards_cpu.mean().item())
                reward_std = float(rewards_cpu.std(unbiased=False).item()) if rewards_cpu.numel() > 1 else 0.0
                metric = {
                    "step": global_step,
                    "algorithm": algo,
                    "source_image_id": source_image_id,
                    "reward_mean": reward_mean,
                    "reward_std": reward_std,
                    "advantage_mean": float(advantages.mean().item()),
                    "lr": float(cfg.train.learning_rate),
                    "kl_coef": float(algo_cfg.kl_coef),
                    "images_per_update": int(rewards_cpu.numel()),
                    "rollout_seconds": rollout_seconds,
                    "reward_seconds": reward_seconds,
                    "update_seconds": update_seconds,
                    "step_seconds": step_seconds,
                    **extra_metric,
                    **last_stats,
                    "details": [asdict(d) for d in details],
                }
                with metrics_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(metric, ensure_ascii=False) + "\n")
                if global_step % int(cfg.train.log_every) == 0 or global_step == 1:
                    log_line = json.dumps(
                        {k: v for k, v in metric.items() if k != "details"},
                        ensure_ascii=False,
                    )
                    if progress_bar is not None:
                        progress_bar.write(log_line)
                    else:
                        print(log_line)
                log_wandb_step(
                    wandb_run,
                    cfg,
                    metric,
                    rewards_cpu,
                    advantages,
                    details,
                    images,
                    device,
                )
                save_training_step_artifacts(
                    output_dir,
                    cfg,
                    global_step,
                    images,
                    batch.get("hq"),
                    rewards_cpu,
                    details,
                )

                if progress_bar is not None:
                    progress_bar.update(1)
                    progress_bar.set_postfix(
                        image=source_image_id,
                        reward=f"{reward_mean:.3f}",
                        generated=int(rewards_cpu.numel()),
                    )

                if global_step % int(cfg.train.ckpt_every) == 0:
                    save_checkpoint(
                        output_dir / "checkpoints" / f"lora_step_{global_step:07d}.pt",
                        cldm,
                        optimizer,
                        global_step,
                        cfg,
                        scaler,
                    )

                eval_cfg = cfg.get("eval", {})
                eval_every = int(eval_cfg.get("every", 0))
                should_eval = (
                    eval_batch is not None
                    and eval_every > 0
                    and global_step % eval_every == 0
                )
                del images, trajectory, cond, uncond, advantages
                if should_eval:
                    release_cuda_cache(device)
                run_fixed_evaluation(
                    cldm,
                    swinir,
                    diffusion,
                    sampler,
                    reward_fn,
                    eval_batch,
                    cfg,
                    algo_cfg,
                    device,
                    global_step,
                    wandb_run,
                    eval_state,
                    eval_metrics_path,
                    output_dir,
                )
                if global_step >= train_steps:
                    break
    finally:
        if progress_bar is not None:
            progress_bar.close()

    save_checkpoint(output_dir / "checkpoints" / "lora_final.pt", cldm, optimizer, global_step, cfg, scaler)
    if wandb_run is not None:
        wandb_run.finish()
    print(f"Done. Output: {output_dir}")


if __name__ == "__main__":
    main()
