from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch
from torch import nn

from diffbir.model.gaussian_diffusion import extract_into_tensor
from diffbir.sampler.spaced_sampler import SpacedSampler


class DDPOSpacedSampler(SpacedSampler):
    """Spaced DDPM sampler with transition log-probabilities for DDPO/PPO."""

    def p_mean_variance(
        self,
        model: nn.Module,
        x: torch.Tensor,
        model_t: torch.Tensor,
        t: torch.Tensor,
        cond: dict[str, torch.Tensor],
        uncond: dict[str, torch.Tensor] | None,
        cfg_scale: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        model_output = self.apply_model(model, x, model_t, cond, uncond, cfg_scale)
        if self.parameterization == "eps":
            pred_x0 = self._predict_xstart_from_eps(x, t, model_output)
        else:
            pred_x0 = self._predict_xstart_from_v(x, t, model_output)
        return self.q_posterior_mean_variance(pred_x0, x, t)

    @staticmethod
    def transition_log_prob(
        x_prev: torch.Tensor,
        mean: torch.Tensor,
        variance: torch.Tensor,
        t: torch.Tensor,
        reduce: str,
    ) -> torch.Tensor:
        variance = variance.clamp_min(1e-20)
        log_prob = -0.5 * (
            (x_prev.float() - mean.float()).pow(2) / variance.float()
            + torch.log(2 * math.pi * variance.float())
        )
        if reduce == "sum":
            log_prob = log_prob.flatten(1).sum(dim=1)
        elif reduce == "mean":
            log_prob = log_prob.flatten(1).mean(dim=1)
        else:
            raise ValueError(f"Unsupported log_prob reduce: {reduce}")
        nonzero = (t != 0).float()
        return log_prob * nonzero

    @staticmethod
    def transition_kl(
        mean: torch.Tensor,
        ref_mean: torch.Tensor,
        variance: torch.Tensor,
        t: torch.Tensor,
        reduce: str,
    ) -> torch.Tensor:
        variance = variance.clamp_min(1e-20)
        kl = 0.5 * (mean.float() - ref_mean.float()).pow(2) / variance.float()
        if reduce == "sum":
            kl = kl.flatten(1).sum(dim=1)
        elif reduce == "mean":
            kl = kl.flatten(1).mean(dim=1)
        else:
            raise ValueError(f"Unsupported KL reduce: {reduce}")
        nonzero = (t != 0).float()
        return kl * nonzero

    def sample_with_log_probs(
        self,
        model: nn.Module,
        device: torch.device,
        steps: int,
        x_size: tuple[int, ...],
        cond: dict[str, torch.Tensor],
        uncond: dict[str, torch.Tensor] | None,
        cfg_scale: float,
        x_T: torch.Tensor | None,
        logprob_reduce: str,
        progress: bool,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        self.make_schedule(steps)
        self.to(device)
        x = x_T if x_T is not None else torch.randn(x_size, device=device, dtype=torch.float32)
        timesteps = np.flip(self.timesteps)
        total_steps = len(self.timesteps)
        bs = x_size[0]

        xs_t: list[torch.Tensor] = []
        xs_prev: list[torch.Tensor] = []
        model_ts: list[torch.Tensor] = []
        ts: list[torch.Tensor] = []
        log_probs: list[torch.Tensor] = []
        masks: list[torch.Tensor] = []
        cfg_scales: list[torch.Tensor] = []

        iterator: Any = timesteps
        if progress:
            from tqdm import tqdm

            iterator = tqdm(timesteps, total=total_steps)

        for i, step in enumerate(iterator):
            model_t = torch.full((bs,), int(step), device=device, dtype=torch.long)
            t = torch.full((bs,), total_steps - i - 1, device=device, dtype=torch.long)
            cur_cfg_scale = self.get_cfg_scale(cfg_scale, int(step))
            mean, variance = self.p_mean_variance(model, x, model_t, t, cond, uncond, cur_cfg_scale)
            noise = torch.randn_like(x)
            nonzero_mask = (t != 0).float().view(-1, *([1] * (x.ndim - 1)))
            x_prev = mean + nonzero_mask * torch.sqrt(variance) * noise
            log_prob = self.transition_log_prob(x_prev, mean, variance, t, logprob_reduce)

            xs_t.append(x.detach().float().cpu())
            xs_prev.append(x_prev.detach().float().cpu())
            model_ts.append(model_t.detach().cpu())
            ts.append(t.detach().cpu())
            log_probs.append(log_prob.detach().cpu())
            masks.append((t != 0).float().detach().cpu())
            cfg_scales.append(torch.full((bs,), float(cur_cfg_scale)).cpu())
            x = x_prev

        trajectory = {
            "x_t": torch.stack(xs_t, dim=1),
            "x_prev": torch.stack(xs_prev, dim=1),
            "model_t": torch.stack(model_ts, dim=1),
            "t": torch.stack(ts, dim=1),
            "old_log_probs": torch.stack(log_probs, dim=1),
            "mask": torch.stack(masks, dim=1),
            "cfg_scale": torch.stack(cfg_scales, dim=1),
        }
        return x, trajectory

    def log_probs_for_trajectory(
        self,
        model: nn.Module,
        trajectory: dict[str, torch.Tensor],
        cond: dict[str, torch.Tensor],
        uncond: dict[str, torch.Tensor] | None,
        device: torch.device,
        logprob_reduce: str,
    ) -> torch.Tensor:
        x_t = trajectory["x_t"].to(device)
        x_prev = trajectory["x_prev"].to(device)
        model_t = trajectory["model_t"].to(device)
        t = trajectory["t"].to(device)
        cfg_scale = trajectory["cfg_scale"].to(device)
        step_log_probs: list[torch.Tensor] = []
        for step_idx in range(x_t.shape[1]):
            mean, variance = self.p_mean_variance(
                model,
                x_t[:, step_idx],
                model_t[:, step_idx],
                t[:, step_idx],
                cond,
                uncond,
                float(cfg_scale[0, step_idx].item()),
            )
            step_log_probs.append(
                self.transition_log_prob(
                    x_prev[:, step_idx],
                    mean,
                    variance,
                    t[:, step_idx],
                    logprob_reduce,
                )
            )
        return torch.stack(step_log_probs, dim=1)

    def reference_kls_for_trajectory(
        self,
        model: nn.Module,
        trajectory: dict[str, torch.Tensor],
        cond: dict[str, torch.Tensor],
        uncond: dict[str, torch.Tensor] | None,
        device: torch.device,
        logprob_reduce: str,
        reference_context: Any,
    ) -> torch.Tensor:
        x_t = trajectory["x_t"].to(device)
        model_t = trajectory["model_t"].to(device)
        t = trajectory["t"].to(device)
        cfg_scale = trajectory["cfg_scale"].to(device)
        step_kls: list[torch.Tensor] = []
        for step_idx in range(x_t.shape[1]):
            mean, variance = self.p_mean_variance(
                model,
                x_t[:, step_idx],
                model_t[:, step_idx],
                t[:, step_idx],
                cond,
                uncond,
                float(cfg_scale[0, step_idx].item()),
            )
            with torch.no_grad():
                with reference_context():
                    ref_mean, _ = self.p_mean_variance(
                        model,
                        x_t[:, step_idx],
                        model_t[:, step_idx],
                        t[:, step_idx],
                        cond,
                        uncond,
                        float(cfg_scale[0, step_idx].item()),
                    )
            step_kls.append(
                self.transition_kl(
                    mean,
                    ref_mean,
                    variance,
                    t[:, step_idx],
                    logprob_reduce,
                )
            )
        return torch.stack(step_kls, dim=1)
