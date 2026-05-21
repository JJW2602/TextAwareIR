#!/usr/bin/env python3
from __future__ import annotations

import argparse
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
        "rl/loss": metric["loss"],
        "rl/policy_loss": metric["policy_loss"],
        "rl/reference_kl": metric["reference_kl"],
        "rl/approx_kl_old_new": metric["approx_kl"],
        "rl/clip_frac": metric["clip_frac"],
        "rl/ratio_mean": metric["ratio_mean"],
        "rl/grad_norm": metric["grad_norm"],
        "rl/lora_param_l2_norm": metric["lora_param_l2_norm"],
        "rl/kl_coef": metric["kl_coef"],
        "system/cuda_memory_allocated_gb": (
            torch.cuda.memory_allocated(device) / (1024**3) if device.type == "cuda" else 0.0
        ),
        "system/cuda_memory_reserved_gb": (
            torch.cuda.memory_reserved(device) / (1024**3) if device.type == "cuda" else 0.0
        ),
    }
    payload.update(tensor_stats("reward", rewards))
    payload.update(tensor_stats("rl/advantage", advantages))
    payload.update(reward_detail_stats(details))

    for key in ("baseline", "group_reward_mean", "group_reward_std", "group_size"):
        if key in metric:
            payload[f"algorithm/{key}"] = metric[key]

    if int(cfg.wandb.log_images_every) > 0 and (
        step == 1 or step % int(cfg.wandb.log_images_every) == 0
    ):
        payload["samples/generated"] = wandb_image_payload(
            images,
            details,
            rewards,
            max_images=int(cfg.wandb.num_log_images),
        )
    if int(cfg.wandb.log_tables_every) > 0 and (
        step == 1 or step % int(cfg.wandb.log_tables_every) == 0
    ):
        payload["samples/reward_details"] = wandb_reward_table(details, rewards)

    wandb_run.log(payload)


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
        progress=bool(cfg.train.progress),
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
    return {
        "image_ids": [f"{base_id}_g{i:02d}" for i in indices],
        "lq": batch["lq"][0:1].repeat(count, 1, 1, 1),
        "gt_instances": [batch["gt_instances"][0] for _ in indices],
        "prompts": [batch["prompts"][0] for _ in indices],
        "negative_prompts": [batch["negative_prompts"][0] for _ in indices],
    }


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
    loss = policy_loss + kl_coef * reference_kl
    clip_frac = (((ratio - 1.0).abs() > clip_range).float() * mask).sum() / denom
    stats = {
        "policy_loss": float(policy_loss.detach().cpu().item()),
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


def save_checkpoint(path: Path, cldm: ControlLDM, optimizer: torch.optim.Optimizer, step: int, cfg: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "step": step,
            "lora": lora_state_dict(cldm),
            "optimizer": optimizer.state_dict(),
            "config": OmegaConf.to_container(cfg, resolve=True),
        },
        path,
    )


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
    wandb_run = init_wandb(cfg, output_dir, algo)

    dataset = SATextParquetDataset(
        parquet_path=cfg.data.parquet,
        manifest_path=cfg.data.get("manifest", None),
        level=int(cfg.data.level),
        start=int(cfg.data.start),
        max_samples=None if cfg.data.max_samples is None else int(cfg.data.max_samples),
        image_size=int(cfg.data.image_size),
        prompt_mode=str(cfg.data.prompt_mode),
        fixed_prompt=str(cfg.data.fixed_prompt),
        negative_prompt=str(cfg.data.negative_prompt),
    )
    loader = DataLoader(
        dataset,
        batch_size=int(cfg.train.batch_size),
        shuffle=True,
        num_workers=int(cfg.train.num_workers),
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_sa_text,
    )
    print(f"Dataset: {len(dataset):,} samples from {dataset.source_description}")

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
    reward_fn = build_reward(cfg)
    sampler = DDPOSpacedSampler(
        diffusion.betas,
        diffusion.parameterization,
        rescale_cfg=bool(cfg.sample.rescale_cfg),
    )
    baseline = 0.0
    global_step = 0
    print(f"Training algorithm: {algo}")

    while global_step < int(cfg.train.train_steps):
        for batch in loader:
            global_step += 1
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
                reward_seconds = time.perf_counter() - reward_start
                advantages, baseline = compute_advantages(rewards_cpu, baseline, cfg)
                extra_metric = {
                    "baseline": baseline,
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
                reward_seconds = time.perf_counter() - reward_start
                advantages, grpo_stats = compute_grpo_advantages(rewards_cpu, cfg)
                extra_metric = {
                    **grpo_stats,
                    "group_size": int(cfg.grpo.group_size),
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
                print(
                    json.dumps(
                        {k: v for k, v in metric.items() if k != "details"},
                        ensure_ascii=False,
                    )
                )
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

            if global_step % int(cfg.train.ckpt_every) == 0:
                save_checkpoint(
                    output_dir / "checkpoints" / f"lora_step_{global_step:07d}.pt",
                    cldm,
                    optimizer,
                    global_step,
                    cfg,
                )
            if global_step >= int(cfg.train.train_steps):
                break

    save_checkpoint(output_dir / "checkpoints" / "lora_final.pt", cldm, optimizer, global_step, cfg)
    if wandb_run is not None:
        wandb_run.finish()
    print(f"Done. Output: {output_dir}")


if __name__ == "__main__":
    main()
