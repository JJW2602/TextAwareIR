#!/usr/bin/env python3
"""Plot TAIR vs DiffBIR per-image reward distributions."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", f"/tmp/matplotlib-{os.getuid()}")
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tair-csv",
        type=Path,
        default=Path(
            "/scratch2/james2602/TextAwareIR/Results/Baselines/Compare/"
            "sa_text_test_lv2_chunks_0_1_tair_vs_diffbir/evaluation/tair/"
            "per_image_reward/per_image_reward.csv"
        ),
    )
    parser.add_argument(
        "--diffbir-csv",
        type=Path,
        default=Path(
            "/scratch2/james2602/TextAwareIR/Results/Baselines/Compare/"
            "sa_text_test_lv2_chunks_0_1_tair_vs_diffbir/evaluation/diffbir/"
            "per_image_reward/per_image_reward.csv"
        ),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(
            "/scratch2/james2602/TextAwareIR/Results/Baselines/Compare/"
            "sa_text_test_lv2_chunks_0_1_tair_vs_diffbir/evaluation/reward_comparison"
        ),
    )
    parser.add_argument("--reward-column", default="final_reward_norm")
    parser.add_argument("--max-images", type=int, default=None)
    return parser.parse_args()


def read_rows(path: Path, max_images: int | None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            parsed: dict[str, Any] = dict(row)
            for key in (
                "num_gt",
                "num_pred",
                "num_matched",
                "missed",
                "false_positive",
            ):
                if key in parsed and parsed[key] != "":
                    parsed[key] = int(float(parsed[key]))
            for key in (
                "matched_mean_reward",
                "final_reward",
                "final_reward_norm",
                "mean_iou_matched",
                "detection_recall",
            ):
                if key in parsed and parsed[key] != "":
                    parsed[key] = float(parsed[key])
            if "false_positive" not in parsed:
                parsed["false_positive"] = max(
                    int(parsed.get("num_pred", 0)) - int(parsed.get("num_matched", 0)),
                    0,
                )
            rows.append(parsed)
            if max_images is not None and len(rows) >= max_images:
                break
    return rows


def describe(values: np.ndarray) -> dict[str, float | int]:
    if values.size == 0:
        return {
            "n": 0,
            "mean": 0.0,
            "std": 0.0,
            "min": 0.0,
            "p25": 0.0,
            "median": 0.0,
            "p75": 0.0,
            "max": 0.0,
        }
    return {
        "n": int(values.size),
        "mean": float(values.mean()),
        "std": float(values.std()),
        "min": float(values.min()),
        "p25": float(np.percentile(values, 25)),
        "median": float(np.percentile(values, 50)),
        "p75": float(np.percentile(values, 75)),
        "max": float(values.max()),
    }


def values(rows: list[dict[str, Any]], column: str) -> np.ndarray:
    return np.asarray([float(row[column]) for row in rows], dtype=float)


def component_means(rows: list[dict[str, Any]]) -> dict[str, float]:
    num_gt = np.asarray([max(int(row["num_gt"]), 1) for row in rows], dtype=float)
    missed = np.asarray([int(row["missed"]) for row in rows], dtype=float)
    false_positive = np.asarray([int(row["false_positive"]) for row in rows], dtype=float)
    return {
        "matched_mean_reward": float(values(rows, "matched_mean_reward").mean()),
        "missed_per_gt": float((missed / num_gt).mean()),
        "false_positive_per_gt": float((false_positive / num_gt).mean()),
        "final_reward_norm": float(values(rows, "final_reward_norm").mean()),
        "final_reward": float(values(rows, "final_reward").mean()),
    }


def write_summary(out_dir: Path, datasets: dict[str, list[dict[str, Any]]], reward_column: str) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for name, rows in datasets.items():
        reward_values = values(rows, reward_column)
        summary[name] = {
            "reward_column": reward_column,
            "reward": describe(reward_values),
            "components": component_means(rows),
        }

    with (out_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    with (out_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "model",
            "reward_column",
            "n",
            "reward_mean",
            "reward_std",
            "reward_median",
            "reward_min",
            "reward_max",
            "matched_mean_reward_mean",
            "missed_per_gt_mean",
            "false_positive_per_gt_mean",
            "final_reward_norm_mean",
            "final_reward_mean",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for name, model_summary in summary.items():
            row = {
                "model": name,
                "reward_column": reward_column,
                "n": model_summary["reward"]["n"],
                "reward_mean": model_summary["reward"]["mean"],
                "reward_std": model_summary["reward"]["std"],
                "reward_median": model_summary["reward"]["median"],
                "reward_min": model_summary["reward"]["min"],
                "reward_max": model_summary["reward"]["max"],
            }
            row.update(
                {
                    "matched_mean_reward_mean": model_summary["components"]["matched_mean_reward"],
                    "missed_per_gt_mean": model_summary["components"]["missed_per_gt"],
                    "false_positive_per_gt_mean": model_summary["components"]["false_positive_per_gt"],
                    "final_reward_norm_mean": model_summary["components"]["final_reward_norm"],
                    "final_reward_mean": model_summary["components"]["final_reward"],
                }
            )
            writer.writerow(row)
    return summary


def plot_distribution(out_dir: Path, datasets: dict[str, list[dict[str, Any]]], reward_column: str) -> None:
    colors = {"TAIR": "#2f7fc1", "DiffBIR": "#d65f5f"}
    all_values = np.concatenate([values(rows, reward_column) for rows in datasets.values()])
    bins = np.linspace(float(all_values.min()), float(all_values.max()), 45)
    if np.allclose(bins[0], bins[-1]):
        bins = np.linspace(bins[0] - 0.5, bins[0] + 0.5, 20)

    fig, ax = plt.subplots(figsize=(8.4, 5.0))
    for name, rows in datasets.items():
        reward_values = values(rows, reward_column)
        ax.hist(
            reward_values,
            bins=bins,
            alpha=0.48,
            density=True,
            color=colors.get(name, None),
            edgecolor="black",
            linewidth=0.4,
            label=f"{name} (mean={reward_values.mean():.3f}, n={reward_values.size})",
        )
        ax.axvline(
            reward_values.mean(),
            color=colors.get(name, None),
            linestyle="--",
            linewidth=2,
        )
    ax.axvline(0.0, color="black", linestyle=":", linewidth=1)
    ax.set_title(f"TAIR vs DiffBIR reward distribution ({reward_column})")
    ax.set_xlabel(reward_column)
    ax.set_ylabel("density")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "reward_distribution_overlay.png", dpi=150)
    plt.close(fig)


def plot_mean_bars(out_dir: Path, datasets: dict[str, list[dict[str, Any]]], reward_column: str) -> None:
    names = list(datasets.keys())
    means = [values(datasets[name], reward_column).mean() for name in names]
    stds = [values(datasets[name], reward_column).std() for name in names]
    colors = ["#2f7fc1", "#d65f5f"]

    fig, ax = plt.subplots(figsize=(5.6, 4.4))
    x = np.arange(len(names))
    ax.bar(x, means, yerr=stds, color=colors[: len(names)], edgecolor="black", capsize=5, alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.axhline(0.0, color="black", linestyle=":", linewidth=1)
    ax.set_ylabel(f"mean {reward_column} (+/- std)")
    ax.set_title("Mean reward over images")
    for idx, mean in enumerate(means):
        ax.text(idx, mean, f"{mean:.3f}", ha="center", va="bottom" if mean >= 0 else "top")
    fig.tight_layout()
    fig.savefig(out_dir / "reward_mean_bar.png", dpi=150)
    plt.close(fig)


def plot_components(out_dir: Path, datasets: dict[str, list[dict[str, Any]]]) -> None:
    names = list(datasets.keys())
    components = [component_means(datasets[name]) for name in names]
    keys = [
        "matched_mean_reward",
        "missed_per_gt",
        "false_positive_per_gt",
        "final_reward_norm",
    ]
    labels = [
        "matched mean",
        "missed / GT",
        "false positive / GT",
        "final reward norm",
    ]

    x = np.arange(len(keys))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    for idx, name in enumerate(names):
        vals = [components[idx][key] for key in keys]
        offset = (idx - (len(names) - 1) / 2.0) * width
        ax.bar(x + offset, vals, width=width, label=name, edgecolor="black", alpha=0.85)
    ax.axhline(0.0, color="black", linestyle=":", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("mean over images")
    ax.set_title("Reward component means")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "reward_component_means.png", dpi=150)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    datasets = {
        "TAIR": read_rows(args.tair_csv, args.max_images),
        "DiffBIR": read_rows(args.diffbir_csv, args.max_images),
    }
    for name, rows in datasets.items():
        if not rows:
            raise SystemExit(f"{name} has no rows.")
        if args.reward_column not in rows[0]:
            raise SystemExit(f"{args.reward_column!r} is missing from {name} rows.")

    summary = write_summary(args.out_dir, datasets, args.reward_column)
    plot_distribution(args.out_dir, datasets, args.reward_column)
    plot_mean_bars(args.out_dir, datasets, args.reward_column)
    plot_components(args.out_dir, datasets)

    for name, model_summary in summary.items():
        reward = model_summary["reward"]
        print(
            f"{name}: n={reward['n']} {args.reward_column} "
            f"mean={reward['mean']:.4f} median={reward['median']:.4f} "
            f"range=[{reward['min']:.4f}, {reward['max']:.4f}]"
        )
    print(f"Output: {args.out_dir}")


if __name__ == "__main__":
    main()
