#!/usr/bin/env python3
"""
Compose a per-image RL reward for DiffBIR text restoration.

For each image with num_gt GT instances and num_matched IoU>=thr matches:

    matched_mean_reward = mean over matched pairs of  max(1 - lev / len(gt_text), 0)
    missed              = num_gt - num_matched
    final_reward        = matched_mean_reward - miss_penalty * missed

We also report a normalized variant:
    final_reward_norm   = matched_mean_reward - (missed / num_gt)

Reads per_image.csv produced by eval_diffbir_text_reward.py and writes a small
summary + plots so the reward signal is easy to inspect.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--per-image-csv",
        type=Path,
        default=Path(
            "/scratch2/james2602/TextAwareIR/Results/Baselines/DiffBIR/sa_text_test_lv2_a6000/"
            "GT_DiffBIR_inference/evaluation/per_image.csv"
        ),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(
            "/scratch2/james2602/TextAwareIR/Results/Baselines/DiffBIR/sa_text_test_lv2_a6000/"
            "GT_DiffBIR_inference/evaluation/per_image_reward"
        ),
    )
    parser.add_argument(
        "--miss-penalty",
        type=float,
        default=1.0,
        help="Penalty per missed GT instance (num_gt - num_matched). Default 1.0.",
    )
    return parser.parse_args()


def read_per_image(csv_path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row["num_gt"] = int(row["num_gt"])
            row["num_pred"] = int(row["num_pred"])
            row["num_matched"] = int(row["num_matched"])
            row["mean_reward_matched"] = float(row["mean_reward_matched"])
            row["mean_iou_matched"] = float(row["mean_iou_matched"])
            row["mean_ned_matched"] = float(row["mean_ned_matched"])
            rows.append(row)
    return rows


def compose_rewards(rows: list[dict[str, Any]], miss_penalty: float) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for r in rows:
        if r["num_gt"] == 0:
            continue
        matched_mean = r["mean_reward_matched"]
        missed = r["num_gt"] - r["num_matched"]
        final = matched_mean - miss_penalty * missed
        final_norm = matched_mean - (missed / r["num_gt"])
        out.append(
            {
                "chunk": r["chunk"],
                "image_id": r["image_id"],
                "num_gt": r["num_gt"],
                "num_pred": r["num_pred"],
                "num_matched": r["num_matched"],
                "missed": missed,
                "matched_mean_reward": matched_mean,
                "final_reward": final,
                "final_reward_norm": final_norm,
                "mean_iou_matched": r["mean_iou_matched"],
                "detection_recall": r["num_matched"] / r["num_gt"],
            }
        )
    return out


def describe(values: list[float] | np.ndarray) -> dict[str, float | int]:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return {k: 0.0 for k in ("n", "mean", "std", "min", "p25", "median", "p75", "max")}
    return {
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "p25": float(np.percentile(arr, 25)),
        "median": float(np.percentile(arr, 50)),
        "p75": float(np.percentile(arr, 75)),
        "max": float(arr.max()),
    }


def bucket_by_gt(rows: list[dict[str, Any]], col: str) -> dict[str, dict[str, float]]:
    """Aggregate `col` over GT-count buckets."""
    buckets = [(1, 1, "1"), (2, 2, "2"), (3, 3, "3"), (4, 5, "4-5"), (6, 10, "6-10"), (11, 9999, "11+")]
    out: dict[str, dict[str, float]] = {}
    for lo, hi, label in buckets:
        values = [r[col] for r in rows if lo <= r["num_gt"] <= hi]
        if not values:
            continue
        arr = np.asarray(values)
        out[label] = {
            "n": int(arr.size),
            "mean": float(arr.mean()),
            "std": float(arr.std()),
            "median": float(np.median(arr)),
        }
    return out


def make_plots(rows: list[dict[str, Any]], out_dir: Path, miss_penalty: float) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    finals = np.array([r["final_reward"] for r in rows])
    finals_norm = np.array([r["final_reward_norm"] for r in rows])
    matched_means = np.array([r["matched_mean_reward"] for r in rows])
    missed = np.array([r["missed"] for r in rows])
    num_gt = np.array([r["num_gt"] for r in rows])
    num_matched = np.array([r["num_matched"] for r in rows])
    det_recall = np.array([r["detection_recall"] for r in rows])

    # 1) Histograms (3 panels in one figure)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    axes[0].hist(matched_means, bins=30, color="#3aa75d", edgecolor="black", alpha=0.85)
    axes[0].set_title("matched_mean_reward\n(positive component)")
    axes[0].set_xlabel("reward")
    axes[0].set_ylabel("# images")
    axes[0].axvline(matched_means.mean(), color="black", linestyle="--", linewidth=1, label=f"mean={matched_means.mean():.3f}")
    axes[0].legend()

    axes[1].hist(missed, bins=range(int(missed.min()), int(missed.max()) + 2),
                 color="#d96157", edgecolor="black", alpha=0.85)
    axes[1].set_title(f"missed instances\n(GT - matched)  penalty/each = {miss_penalty}")
    axes[1].set_xlabel("# missed")
    axes[1].set_ylabel("# images")
    axes[1].axvline(missed.mean(), color="black", linestyle="--", linewidth=1, label=f"mean={missed.mean():.2f}")
    axes[1].legend()

    axes[2].hist(finals, bins=40, color="#3b78c6", edgecolor="black", alpha=0.85)
    axes[2].set_title("final_reward\n(matched_mean - penalty × missed)")
    axes[2].set_xlabel("reward")
    axes[2].set_ylabel("# images")
    axes[2].axvline(finals.mean(), color="black", linestyle="--", linewidth=1, label=f"mean={finals.mean():.3f}")
    axes[2].axvline(0.0, color="red", linestyle=":", linewidth=1, label="r=0")
    axes[2].legend()

    fig.suptitle(
        f"Per-image reward components  (N={len(rows)} images,  miss_penalty={miss_penalty})",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(out_dir / "reward_components_hist.png", dpi=130)
    plt.close(fig)

    # 2) Final-reward and normalized comparison
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].hist(finals, bins=40, color="#3b78c6", edgecolor="black", alpha=0.85)
    axes[0].set_title(f"final_reward = matched_mean - {miss_penalty} × missed")
    axes[0].set_xlabel("reward")
    axes[0].set_ylabel("# images")
    axes[0].axvline(finals.mean(), color="black", linestyle="--", linewidth=1, label=f"mean={finals.mean():.3f}")
    axes[0].axvline(0.0, color="red", linestyle=":", linewidth=1)
    axes[0].legend()

    axes[1].hist(finals_norm, bins=40, color="#7a59c0", edgecolor="black", alpha=0.85)
    axes[1].set_title("final_reward_norm = matched_mean - missed / num_gt\n(bounded in [-1, 1])")
    axes[1].set_xlabel("reward")
    axes[1].set_ylabel("# images")
    axes[1].axvline(finals_norm.mean(), color="black", linestyle="--", linewidth=1, label=f"mean={finals_norm.mean():.3f}")
    axes[1].axvline(0.0, color="red", linestyle=":", linewidth=1)
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(out_dir / "final_reward_hist.png", dpi=130)
    plt.close(fig)

    # 3) Scatter — num_gt vs final_reward, color=detection_recall
    fig, ax = plt.subplots(figsize=(7, 5))
    sc = ax.scatter(num_gt, finals, c=det_recall, cmap="viridis", s=14, alpha=0.7)
    ax.set_xlabel("# GT instances in image")
    ax.set_ylabel("final_reward")
    ax.set_title("final_reward vs GT count (color = detection recall)")
    ax.axhline(0.0, color="red", linestyle=":", linewidth=1)
    cb = fig.colorbar(sc, ax=ax)
    cb.set_label("detection recall (matched / num_gt)")
    fig.tight_layout()
    fig.savefig(out_dir / "scatter_numgt_vs_final.png", dpi=130)
    plt.close(fig)

    # 4) Reward by GT-count bucket (bar)
    buckets = bucket_by_gt(rows, "final_reward")
    if buckets:
        labels = list(buckets.keys())
        means = [buckets[k]["mean"] for k in labels]
        stds = [buckets[k]["std"] for k in labels]
        ns = [buckets[k]["n"] for k in labels]
        fig, ax = plt.subplots(figsize=(7, 4.2))
        x = np.arange(len(labels))
        ax.bar(x, means, yerr=stds, color="#3b78c6", edgecolor="black", capsize=4, alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{lab}\nn={n}" for lab, n in zip(labels, ns)])
        ax.set_ylabel("mean final_reward (±std)")
        ax.set_xlabel("# GT instances bucket")
        ax.set_title("Mean final_reward by GT-count bucket")
        ax.axhline(0.0, color="red", linestyle=":", linewidth=1)
        fig.tight_layout()
        fig.savefig(out_dir / "bucket_by_numgt.png", dpi=130)
        plt.close(fig)

    # 5) Matched mean vs missed (2D heat)
    fig, ax = plt.subplots(figsize=(7, 5))
    hb = ax.hexbin(missed, matched_means, gridsize=20, cmap="magma", mincnt=1)
    ax.set_xlabel("# missed GT instances")
    ax.set_ylabel("matched_mean_reward")
    ax.set_title("Where images sit on (missed count, matched-mean reward)")
    cb = fig.colorbar(hb, ax=ax)
    cb.set_label("# images")
    fig.tight_layout()
    fig.savefig(out_dir / "hexbin_missed_vs_matched_mean.png", dpi=130)
    plt.close(fig)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = read_per_image(args.per_image_csv)
    composed = compose_rewards(rows, args.miss_penalty)

    finals = [r["final_reward"] for r in composed]
    finals_norm = [r["final_reward_norm"] for r in composed]
    matched_means = [r["matched_mean_reward"] for r in composed]
    missed = [r["missed"] for r in composed]

    summary = {
        "input_csv": str(args.per_image_csv),
        "num_images_with_gt": len(composed),
        "miss_penalty": args.miss_penalty,
        "reward_definition": (
            "final_reward = mean_{matched pairs}(max(1 - lev / len(gt_text), 0)) "
            f"- {args.miss_penalty} * (num_gt - num_matched)"
        ),
        "reward_definition_norm": (
            "final_reward_norm = matched_mean_reward - (num_gt - num_matched) / num_gt"
        ),
        "matched_mean_reward": describe(matched_means),
        "missed_count": describe(missed),
        "final_reward": describe(finals),
        "final_reward_norm": describe(finals_norm),
        "frac_final_positive": float(np.mean(np.array(finals) > 0)) if finals else 0.0,
        "frac_final_zero": float(np.mean(np.array(finals) == 0)) if finals else 0.0,
        "frac_final_negative": float(np.mean(np.array(finals) < 0)) if finals else 0.0,
        "buckets_by_num_gt": {
            "final_reward": bucket_by_gt(composed, "final_reward"),
            "matched_mean_reward": bucket_by_gt(composed, "matched_mean_reward"),
            "detection_recall": bucket_by_gt(composed, "detection_recall"),
        },
    }

    write_csv(args.out_dir / "per_image_reward.csv", composed)
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    make_plots(composed, args.out_dir, args.miss_penalty)

    print(f"N images with GT: {len(composed)}")
    print(f"final_reward      mean={summary['final_reward']['mean']:.3f}  median={summary['final_reward']['median']:.3f}  range=[{summary['final_reward']['min']:.2f}, {summary['final_reward']['max']:.2f}]")
    print(f"final_reward_norm mean={summary['final_reward_norm']['mean']:.3f}  median={summary['final_reward_norm']['median']:.3f}  range=[{summary['final_reward_norm']['min']:.2f}, {summary['final_reward_norm']['max']:.2f}]")
    print(f"fraction positive / zero / negative: {summary['frac_final_positive']:.2%} / {summary['frac_final_zero']:.2%} / {summary['frac_final_negative']:.2%}")
    print(f"Output: {args.out_dir}")


if __name__ == "__main__":
    main()
