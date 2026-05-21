from __future__ import annotations

import json
import re
import shutil
import sys
import unicodedata
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from .data import TextInstance


@dataclass(frozen=True)
class PredictionInstance:
    text: str
    bbox: tuple[float, float, float, float]
    score: float | None = None


@dataclass(frozen=True)
class RewardDetail:
    image_id: str
    num_gt: int
    num_pred: int
    num_matched: int
    missed: int
    false_positive: int
    matched_mean_reward: float
    final_reward: float
    final_reward_norm: float
    mean_iou_matched: float


def normalize_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text or "")
    return re.sub(r"\s+", " ", normalized).strip()


def levenshtein(left: str, right: str) -> int:
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
            current.append(
                min(
                    current[j - 1] + 1,
                    previous[j] + 1,
                    previous[j - 1] + (left_char != right_char),
                )
            )
        previous = current
    return previous[-1]


def bbox_iou(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
) -> float:
    lx1, ly1, lx2, ly2 = left
    rx1, ry1, rx2, ry2 = right
    ix1, iy1 = max(lx1, rx1), max(ly1, ry1)
    ix2, iy2 = min(lx2, rx2), min(ly2, ry2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    left_area = max(0.0, lx2 - lx1) * max(0.0, ly2 - ly1)
    right_area = max(0.0, rx2 - rx1) * max(0.0, ry2 - ry1)
    union = left_area + right_area - inter
    return inter / union if union > 0 else 0.0


def compose_image_reward(
    image_id: str,
    gt_instances: list[TextInstance],
    pred_instances: list[PredictionInstance],
    iou_threshold: float,
    miss_penalty: float,
    false_positive_penalty: float,
) -> RewardDetail:
    candidates: list[tuple[float, int, int]] = []
    for gi, gt in enumerate(gt_instances):
        for pi, pred in enumerate(pred_instances):
            iou = bbox_iou(gt.bbox, pred.bbox)
            if iou >= iou_threshold:
                candidates.append((iou, gi, pi))
    candidates.sort(reverse=True)

    matched_gt: set[int] = set()
    matched_pred: set[int] = set()
    rewards: list[float] = []
    ious: list[float] = []
    for iou, gi, pi in candidates:
        if gi in matched_gt or pi in matched_pred:
            continue
        gt_text = normalize_text(gt_instances[gi].text)
        pred_text = normalize_text(pred_instances[pi].text)
        dist = levenshtein(gt_text, pred_text)
        rewards.append(max(1.0 - dist / max(len(gt_text), 1), 0.0))
        ious.append(iou)
        matched_gt.add(gi)
        matched_pred.add(pi)

    num_gt = len(gt_instances)
    num_pred = len(pred_instances)
    num_matched = len(rewards)
    missed = max(num_gt - num_matched, 0)
    false_positive = max(num_pred - num_matched, 0)
    matched_mean = float(np.mean(rewards)) if rewards else 0.0
    final_reward = (
        matched_mean
        - miss_penalty * missed
        - false_positive_penalty * false_positive
    )
    norm_denominator = max(num_gt, 1)
    final_reward_norm = (
        matched_mean
        - miss_penalty * missed / norm_denominator
        - false_positive_penalty * false_positive / norm_denominator
    )
    return RewardDetail(
        image_id=image_id,
        num_gt=num_gt,
        num_pred=num_pred,
        num_matched=num_matched,
        missed=missed,
        false_positive=false_positive,
        matched_mean_reward=matched_mean,
        final_reward=final_reward,
        final_reward_norm=final_reward_norm,
        mean_iou_matched=float(np.mean(ious)) if ious else 0.0,
    )


def load_bridge_predictions(json_path: Path, score_threshold: float) -> dict[str, list[PredictionInstance]]:
    if not json_path.is_file():
        return {}
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    out: dict[str, list[PredictionInstance]] = {}
    for ann in payload.get("annotations", []):
        file_name = ann.get("file_name")
        bbox = ann.get("bbox")
        if not file_name or not (isinstance(bbox, (list, tuple)) and len(bbox) == 4):
            continue
        score = float(ann.get("score", 0.0))
        if score < score_threshold:
            continue
        text = ann.get("rec", ann.get("text", ann.get("VLM", "")))
        out.setdefault(Path(file_name).stem, []).append(
            PredictionInstance(
                text=str(text or ""),
                bbox=tuple(float(v) for v in bbox),
                score=score,
            )
        )
    return out


def save_tensor_images(image_ids: list[str], images: torch.Tensor, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    arr = (
        images.detach()
        .float()
        .clamp(0, 1)
        .mul(255)
        .byte()
        .permute(0, 2, 3, 1)
        .cpu()
        .numpy()
    )
    for image_id, image_arr in zip(image_ids, arr):
        Image.fromarray(image_arr).save(out_dir / f"{image_id}.png")


class BridgeReward:
    """Bridge text-spotting reward with missed and false-positive penalties."""

    def __init__(
        self,
        config_path: str | Path,
        pipeline_root: str | Path,
        work_dir: str | Path,
        iou_threshold: float,
        bridge_score_threshold: float,
        miss_penalty: float,
        false_positive_penalty: float,
        variant: str,
        keep_images_every: int,
        bridge_env_python: str,
    ) -> None:
        self.config_path = Path(config_path)
        self.pipeline_root = Path(pipeline_root)
        self.work_dir = Path(work_dir)
        self.iou_threshold = iou_threshold
        self.bridge_score_threshold = bridge_score_threshold
        self.miss_penalty = miss_penalty
        self.false_positive_penalty = false_positive_penalty
        self.variant = variant
        self.keep_images_every = keep_images_every
        self.bridge_runner, self.filtering = self._import_pipeline_modules()
        self.config = self._load_yaml(self.config_path)
        if bridge_env_python:
            self.config["bridge_env_python"] = bridge_env_python
        self.work_dir.mkdir(parents=True, exist_ok=True)

    def _import_pipeline_modules(self) -> tuple[Any, Any]:
        curation_root = self.pipeline_root / "dataset_curation"
        if not curation_root.is_dir():
            raise FileNotFoundError(f"dataset_curation directory not found: {curation_root}")
        sys.path.insert(0, str(curation_root))
        from src import bridge_runner, filtering

        return bridge_runner, filtering

    @staticmethod
    def _load_yaml(path: Path) -> dict[str, Any]:
        try:
            import yaml
        except ImportError as exc:
            raise RuntimeError("PyYAML is required for Bridge reward config loading.") from exc
        with path.open("r", encoding="utf-8") as handle:
            return yaml.safe_load(handle)

    def __call__(
        self,
        image_ids: list[str],
        images: torch.Tensor,
        gt_instances: list[list[TextInstance]],
        step: int,
    ) -> tuple[torch.Tensor, list[RewardDetail]]:
        step_dir = self.work_dir / f"step_{step:07d}"
        image_dir = step_dir / "images"
        bridge_dir = step_dir / "bridge"
        filtered_json = step_dir / "bridge_filtered.json"
        save_tensor_images(image_ids, images, image_dir)

        raw_json = self.bridge_runner.run_bridge(
            config=self.config,
            input_dir=str(image_dir),
            output_dir=str(bridge_dir),
            stage1=False,
        )
        if raw_json is None:
            raise RuntimeError("Bridge reward inference failed.")
        if self.filtering.filter_duplicate_detections(
            raw_json,
            str(filtered_json),
            self.config,
        ) is None:
            raise RuntimeError("Bridge duplicate filtering failed.")

        preds = load_bridge_predictions(filtered_json, self.bridge_score_threshold)
        details = [
            compose_image_reward(
                image_id=image_id,
                gt_instances=gt,
                pred_instances=preds.get(image_id, []),
                iou_threshold=self.iou_threshold,
                miss_penalty=self.miss_penalty,
                false_positive_penalty=self.false_positive_penalty,
            )
            for image_id, gt in zip(image_ids, gt_instances)
        ]
        rewards = []
        for detail in details:
            if self.variant == "final_reward":
                rewards.append(detail.final_reward)
            elif self.variant == "final_reward_norm":
                rewards.append(detail.final_reward_norm)
            else:
                raise ValueError(f"Unsupported reward variant: {self.variant}")

        if self.keep_images_every <= 0 or step % self.keep_images_every != 0:
            shutil.rmtree(step_dir, ignore_errors=True)
        else:
            (step_dir / "reward_details.json").write_text(
                json.dumps([asdict(d) for d in details], indent=2),
                encoding="utf-8",
            )

        return torch.tensor(rewards, dtype=torch.float32), details


class ConstantReward:
    """Debug-only reward backend for checking the DDPO loop without Bridge."""

    def __init__(self, value: float, variant: str = "final_reward_norm") -> None:
        self.value = float(value)
        self.variant = variant

    def __call__(
        self,
        image_ids: list[str],
        images: torch.Tensor,
        gt_instances: list[list[TextInstance]],
        step: int,
    ) -> tuple[torch.Tensor, list[RewardDetail]]:
        details = [
            RewardDetail(
                image_id=image_id,
                num_gt=len(gt),
                num_pred=0,
                num_matched=0,
                missed=len(gt),
                false_positive=0,
                matched_mean_reward=0.0,
                final_reward=self.value,
                final_reward_norm=self.value,
                mean_iou_matched=0.0,
            )
            for image_id, gt in zip(image_ids, gt_instances)
        ]
        return torch.full((len(image_ids),), self.value, dtype=torch.float32), details
