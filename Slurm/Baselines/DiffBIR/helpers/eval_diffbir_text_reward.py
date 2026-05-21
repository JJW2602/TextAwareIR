#!/usr/bin/env python3
"""
Evaluate DiffBIR text annotations (Bridge spotter predictions) against SA-Text
GT and compute Flow-GRPO style edit-distance reward.

Outputs:
  <out-dir>/summary.md           Human-readable markdown report
  <out-dir>/summary.json         Full numeric summary
  <out-dir>/per_image.csv        Per-image counts + reward + IoU stats
  <out-dir>/matched_pairs.csv    Every IoU>=thr pair (gt_text, pred_text, NED, reward)
  <out-dir>/unmatched_gt.csv     GT instances with no IoU>=thr prediction
  <out-dir>/unmatched_pred.csv   Predictions with no IoU>=thr GT

Metrics
=======
1. Instance counts (GT vs pred per image): mean / std / median / quantiles / min / max
2. Detection @ IoU>=thr: precision / recall / F1, IoU distribution histogram
3. Recognition on matched pairs:
      - exact match accuracy (case-sensitive / case-insensitive)
      - mean NED  = 1 - lev / max(len_gt, len_pred)
4. End-to-end exact (IoU>=thr AND exact text): precision / recall / F1
5. Flow-GRPO reward on matched pairs:
      r_i = max(1 - lev(pred_rec_i, gt_text_i) / max(len(gt_text_i), 1), 0)
      We report mean / std / median / quantile breakdown AND the per-image mean.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import re
import statistics
import sys
import unicodedata
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable


# ----------------------------- data classes ---------------------------------


@dataclass(frozen=True)
class Instance:
    image_id: str
    text: str
    bbox: tuple[float, float, float, float]
    source: str  # "gt" or "pred"
    score: float | None = None


@dataclass(frozen=True)
class Match:
    image_id: str
    gt_index: int
    pred_index: int
    iou: float
    gt_text: str
    pred_text: str
    gt_text_norm: str
    pred_text_norm: str
    lev_distance: int
    ned: float           # 1 - lev / max(len_gt, len_pred)
    reward: float        # max(1 - lev / max(len_gt, 1), 0)   <-- Flow-GRPO style
    exact_match: bool
    exact_match_ci: bool


# ----------------------------- IO helpers -----------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--gt-parquet",
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
        help="Run root that contains chunk_*/ and text_annotations/.",
    )
    parser.add_argument("--chunks", nargs="+", default=["chunk_0", "chunk_1"])
    parser.add_argument(
        "--bridge-json-fmt",
        type=str,
        default="{results_root}/text_annotations/{chunk}/pipeline/bridge_filtered.json",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(
            "/scratch2/james2602/TextAwareIR/Results/Baselines/DiffBIR/"
            "sa_text_test_lv2_a6000/GT_DiffBIR_inference/evaluation"
        ),
    )
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument(
        "--bridge-score-threshold",
        type=float,
        default=0.0,
        help="Drop Bridge predictions below this confidence before matching.",
    )
    parser.add_argument(
        "--image-extension",
        default=".png",
        help="Extension of restored images sitting at {results_root}/{chunk}/.",
    )
    return parser.parse_args()


def require_pyarrow() -> Any:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise SystemExit("This script needs pyarrow. Install in the DiffBIR env.") from exc
    return pq


def safe_stem(value: str) -> str:
    return Path(str(value)).stem


def normalize_text(text: str) -> str:
    n = unicodedata.normalize("NFKC", text or "")
    return re.sub(r"\s+", " ", n).strip()


def levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        curr = [i]
        for j, cb in enumerate(b, start=1):
            ins = curr[j - 1] + 1
            dele = prev[j] + 1
            sub = prev[j - 1] + (ca != cb)
            curr.append(min(ins, dele, sub))
        prev = curr
    return prev[-1]


def bbox_iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    aa = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    bb = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = aa + bb - inter
    return inter / union if union > 0 else 0.0


# ----------------------------- loading --------------------------------------


def load_gt(parquet_path: Path) -> dict[str, list[Instance]]:
    pq = require_pyarrow()
    table = pq.read_table(parquet_path, columns=["id", "text", "bbox"])
    out: dict[str, list[Instance]] = {}
    for row in table.to_pylist():
        image_id = safe_stem(str(row["id"]))
        texts = row["text"] or []
        bboxes = row["bbox"] or []
        instances: list[Instance] = []
        for t, b in zip(texts, bboxes):
            if not (isinstance(b, list) and len(b) == 2 and len(b[0]) == 2 and len(b[1]) == 2):
                continue
            x1, y1 = b[0]
            x2, y2 = b[1]
            instances.append(
                Instance(
                    image_id=image_id,
                    text=str(t),
                    bbox=(float(x1), float(y1), float(x2), float(y2)),
                    source="gt",
                )
            )
        out[image_id] = instances
    return out


def load_bridge_preds(json_path: Path, score_threshold: float) -> dict[str, list[Instance]]:
    if not json_path.is_file():
        return {}
    with json_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    anns = payload.get("annotations") if isinstance(payload, dict) else None
    if not anns:
        return {}
    out: dict[str, list[Instance]] = {}
    for ann in anns:
        file_name = ann.get("file_name")
        if not file_name:
            continue
        score = float(ann.get("score", 0.0))
        if score < score_threshold:
            continue
        bbox = ann.get("bbox")
        if not (isinstance(bbox, (list, tuple)) and len(bbox) == 4):
            continue
        x1, y1, x2, y2 = (float(v) for v in bbox)
        image_id = safe_stem(file_name)
        out.setdefault(image_id, []).append(
            Instance(
                image_id=image_id,
                text=str(ann.get("rec", "") or ""),
                bbox=(x1, y1, x2, y2),
                source="pred",
                score=score,
            )
        )
    return out


def list_chunk_image_ids(results_root: Path, chunk: str, ext: str) -> set[str]:
    chunk_dir = results_root / chunk
    if not chunk_dir.is_dir():
        return set()
    return {p.stem for p in chunk_dir.glob(f"*{ext}")}


# ----------------------------- matching -------------------------------------


def match_image(
    gt_instances: list[Instance],
    pred_instances: list[Instance],
    iou_threshold: float,
) -> tuple[list[Match], list[int], list[int]]:
    candidates: list[tuple[float, int, int]] = []
    for gi, g in enumerate(gt_instances):
        for pi, p in enumerate(pred_instances):
            iou = bbox_iou(g.bbox, p.bbox)
            if iou >= iou_threshold:
                candidates.append((iou, gi, pi))
    candidates.sort(reverse=True)

    matched_g: set[int] = set()
    matched_p: set[int] = set()
    matches: list[Match] = []
    for iou, gi, pi in candidates:
        if gi in matched_g or pi in matched_p:
            continue
        g = gt_instances[gi]
        p = pred_instances[pi]
        gt_norm = normalize_text(g.text)
        pred_norm = normalize_text(p.text)
        lev = levenshtein(gt_norm, pred_norm)
        max_len = max(len(gt_norm), len(pred_norm), 1)
        ned = 1.0 - lev / max_len
        ref_len = max(len(gt_norm), 1)
        reward = max(1.0 - lev / ref_len, 0.0)
        matches.append(
            Match(
                image_id=g.image_id,
                gt_index=gi,
                pred_index=pi,
                iou=iou,
                gt_text=g.text,
                pred_text=p.text,
                gt_text_norm=gt_norm,
                pred_text_norm=pred_norm,
                lev_distance=lev,
                ned=ned,
                reward=reward,
                exact_match=gt_norm == pred_norm,
                exact_match_ci=gt_norm.casefold() == pred_norm.casefold(),
            )
        )
        matched_g.add(gi)
        matched_p.add(pi)
    unmatched_g = [i for i in range(len(gt_instances)) if i not in matched_g]
    unmatched_p = [i for i in range(len(pred_instances)) if i not in matched_p]
    return matches, unmatched_g, unmatched_p


# ----------------------------- aggregation ----------------------------------


def describe(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {"n": 0, "mean": 0.0, "std": 0.0, "min": 0.0, "p25": 0.0, "median": 0.0, "p75": 0.0, "max": 0.0}
    n = len(values)
    mean = statistics.fmean(values)
    std = statistics.pstdev(values) if n > 1 else 0.0
    sorted_v = sorted(values)
    if n >= 4:
        q = statistics.quantiles(values, n=4, method="inclusive")
        p25, median, p75 = q[0], q[1], q[2]
    else:
        p25 = sorted_v[0]
        median = sorted_v[n // 2]
        p75 = sorted_v[-1]
    return {
        "n": n,
        "mean": mean,
        "std": std,
        "min": sorted_v[0],
        "p25": p25,
        "median": median,
        "p75": p75,
        "max": sorted_v[-1],
    }


def iou_histogram(matches: list[Match], thr: float) -> dict[str, int]:
    bins = [(thr, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.0001)]
    labels = ["[thr,0.6)", "[0.6,0.7)", "[0.7,0.8)", "[0.8,0.9)", "[0.9,1.0]"]
    counts = [0] * len(bins)
    for m in matches:
        for i, (lo, hi) in enumerate(bins):
            if lo <= m.iou < hi:
                counts[i] += 1
                break
    return dict(zip(labels, counts))


def reward_histogram(matches: list[Match]) -> dict[str, int]:
    bins = [(0.0, 0.0001), (0.0001, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 0.9999), (0.9999, 1.01)]
    labels = ["=0.00", "(0,0.25)", "[0.25,0.5)", "[0.5,0.75)", "[0.75,1)", "=1.00"]
    counts = [0] * len(bins)
    for m in matches:
        for i, (lo, hi) in enumerate(bins):
            if lo <= m.reward < hi:
                counts[i] += 1
                break
    return dict(zip(labels, counts))


def safe_div(a: float, b: float) -> float:
    return a / b if b else 0.0


def f1(p: float, r: float) -> float:
    return 2 * p * r / (p + r) if (p + r) else 0.0


# ----------------------------- formatting -----------------------------------


def fmt_pct(x: float) -> str:
    return f"{100.0 * x:.2f}%"


def fmt_describe(d: dict[str, float | int]) -> str:
    return (
        f"n={d['n']}, mean={d['mean']:.3f}, std={d['std']:.3f}, "
        f"min={d['min']:.3f}, p25={d['p25']:.3f}, median={d['median']:.3f}, p75={d['p75']:.3f}, max={d['max']:.3f}"
    )


def render_markdown(summary: dict[str, Any]) -> str:
    lines: list[str] = []
    L = lines.append
    L("# DiffBIR Text Annotation Evaluation")
    L("")
    L(f"- **GT parquet**: `{summary['gt_parquet']}`")
    L(f"- **Results root**: `{summary['results_root']}`")
    L(f"- **Chunks**: {', '.join(summary['chunks'])}")
    L(f"- **IoU threshold**: {summary['iou_threshold']}")
    L(f"- **Bridge score threshold**: {summary['bridge_score_threshold']}")
    L("")

    # Counts
    counts = summary["counts"]
    L("## 1. Instance Counts")
    L("")
    L("| | value |")
    L("|---|---:|")
    L(f"| Images evaluated | {counts['num_images']} |")
    L(f"| Images with at least one GT | {counts['num_images_with_gt']} |")
    L(f"| Images with at least one pred | {counts['num_images_with_pred']} |")
    L(f"| Total GT instances | {counts['total_gt']} |")
    L(f"| Total Bridge predictions | {counts['total_pred']} |")
    L(f"| Pred / GT ratio | {counts['pred_per_gt_ratio']:.3f} |")
    L("")
    L("### Per-image distribution")
    L("")
    L("| | n | mean | std | min | p25 | median | p75 | max |")
    L("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for name in ["gt_per_image", "pred_per_image", "pred_minus_gt_per_image"]:
        d = counts[name]
        L(f"| {name} | {d['n']} | {d['mean']:.2f} | {d['std']:.2f} | {d['min']:.0f} | {d['p25']:.1f} | {d['median']:.1f} | {d['p75']:.1f} | {d['max']:.0f} |")
    L("")

    # Detection
    det = summary["detection"]
    L(f"## 2. Detection @ IoU >= {summary['iou_threshold']}")
    L("")
    L("| metric | value |")
    L("|---|---:|")
    L(f"| Matched pairs | {det['num_matches']} |")
    L(f"| Precision (matches / pred) | {fmt_pct(det['precision'])} |")
    L(f"| Recall (matches / gt) | {fmt_pct(det['recall'])} |")
    L(f"| F1 | {fmt_pct(det['f1'])} |")
    L(f"| Mean IoU on matched | {det['mean_iou']:.3f} |")
    L("")
    L("### IoU distribution (matched pairs)")
    L("")
    L("| bin | count |")
    L("|---|---:|")
    for k, v in det["iou_histogram"].items():
        L(f"| {k} | {v} |")
    L("")

    # Recognition
    rec = summary["recognition_on_matched"]
    L("## 3. Recognition (on IoU-matched pairs)")
    L("")
    L("| metric | value |")
    L("|---|---:|")
    L(f"| Exact match (case-sensitive) | {fmt_pct(rec['exact_match_acc'])} |")
    L(f"| Exact match (case-insensitive) | {fmt_pct(rec['exact_match_acc_ci'])} |")
    L(f"| Mean NED  (1 - lev / max(len)) | {rec['mean_ned']:.4f} |")
    L("")

    # End-to-end
    e2e = summary["end_to_end_exact"]
    L(f"## 4. End-to-End Exact (IoU>=thr AND text exact, case-sensitive)")
    L("")
    L("| metric | value |")
    L("|---|---:|")
    L(f"| Precision | {fmt_pct(e2e['precision'])} |")
    L(f"| Recall | {fmt_pct(e2e['recall'])} |")
    L(f"| F1 | {fmt_pct(e2e['f1'])} |")
    L("")

    # Flow-GRPO reward
    rw = summary["flow_grpo_reward"]
    L("## 5. Flow-GRPO Style Reward")
    L("")
    L("Per matched pair (IoU >= thr):")
    L("")
    L("```")
    L("    r_i = max(1 - lev(pred_rec_i, gt_text_i) / max(len(gt_text_i), 1), 0)")
    L("```")
    L("")
    L("| aggregation | value |")
    L("|---|---:|")
    L(f"| Mean over matched pairs | {rw['mean_over_matched']:.4f} |")
    L(f"| Std  over matched pairs | {rw['std_over_matched']:.4f} |")
    L(f"| Median | {rw['median_over_matched']:.4f} |")
    L(f"| Fraction with r = 1.0 (perfect) | {fmt_pct(rw['frac_perfect'])} |")
    L(f"| Fraction with r = 0.0 (worse than reference length) | {fmt_pct(rw['frac_zero'])} |")
    L(f"| Mean per-image reward (matched-only avg, then avg over images that have matches) | {rw['mean_per_image_matched_only']:.4f} |")
    L(f"| Mean per-image reward (sum / num_gt, unmatched GT counted as 0) | {rw['mean_per_image_over_gt']:.4f} |")
    L("")
    L("### Reward distribution (matched pairs)")
    L("")
    L("| bin | count |")
    L("|---|---:|")
    for k, v in rw["histogram"].items():
        L(f"| {k} | {v} |")
    L("")

    # Per-chunk
    if summary.get("per_chunk"):
        L("## 6. Per-chunk Breakdown")
        L("")
        L("| chunk | images | GT | pred | matched | det F1 | mean NED | mean reward |")
        L("|---|---:|---:|---:|---:|---:|---:|---:|")
        for chunk, c in summary["per_chunk"].items():
            L(
                f"| {chunk} | {c['num_images']} | {c['total_gt']} | {c['total_pred']} "
                f"| {c['num_matches']} | {fmt_pct(c['detection_f1'])} | {c['mean_ned']:.4f} "
                f"| {c['mean_reward']:.4f} |"
            )
        L("")

    return "\n".join(lines) + "\n"


# ----------------------------- main -----------------------------------------


def evaluate_chunk(
    chunk: str,
    image_ids: set[str],
    gt_all: dict[str, list[Instance]],
    pred_chunk: dict[str, list[Instance]],
    iou_threshold: float,
) -> tuple[list[Match], list[dict[str, Any]], list[Instance], list[Instance]]:
    matches: list[Match] = []
    per_image_rows: list[dict[str, Any]] = []
    unmatched_gt: list[Instance] = []
    unmatched_pred: list[Instance] = []
    for image_id in sorted(image_ids):
        g = gt_all.get(image_id, [])
        p = pred_chunk.get(image_id, [])
        m, ug, up = match_image(g, p, iou_threshold)
        matches.extend(m)
        unmatched_gt.extend(g[i] for i in ug)
        unmatched_pred.extend(p[i] for i in up)
        per_image_rows.append(
            {
                "chunk": chunk,
                "image_id": image_id,
                "num_gt": len(g),
                "num_pred": len(p),
                "num_matched": len(m),
                "mean_iou_matched": statistics.fmean(x.iou for x in m) if m else 0.0,
                "exact_matches": sum(x.exact_match for x in m),
                "mean_ned_matched": statistics.fmean(x.ned for x in m) if m else 0.0,
                "mean_reward_matched": statistics.fmean(x.reward for x in m) if m else 0.0,
                "reward_sum_over_gt": (sum(x.reward for x in m) / len(g)) if g else 0.0,
            }
        )
    return matches, per_image_rows, unmatched_gt, unmatched_pred


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

    gt_all = load_gt(args.gt_parquet)

    all_matches: list[Match] = []
    all_per_image: list[dict[str, Any]] = []
    all_unmatched_gt: list[Instance] = []
    all_unmatched_pred: list[Instance] = []
    per_chunk_summary: dict[str, dict[str, Any]] = {}

    total_images = 0
    total_gt = 0
    total_pred = 0

    for chunk in args.chunks:
        chunk_image_ids = list_chunk_image_ids(args.results_root, chunk, args.image_extension)
        bridge_path = Path(args.bridge_json_fmt.format(results_root=str(args.results_root), chunk=chunk))
        pred_chunk = load_bridge_preds(bridge_path, args.bridge_score_threshold)
        matches, per_image, ug, up = evaluate_chunk(
            chunk, chunk_image_ids, gt_all, pred_chunk, args.iou_threshold
        )
        all_matches.extend(matches)
        all_per_image.extend(per_image)
        all_unmatched_gt.extend(ug)
        all_unmatched_pred.extend(up)

        c_gt = sum(r["num_gt"] for r in per_image)
        c_pred = sum(r["num_pred"] for r in per_image)
        per_chunk_summary[chunk] = {
            "num_images": len(per_image),
            "total_gt": c_gt,
            "total_pred": c_pred,
            "num_matches": len(matches),
            "detection_precision": safe_div(len(matches), c_pred),
            "detection_recall": safe_div(len(matches), c_gt),
            "detection_f1": f1(safe_div(len(matches), c_pred), safe_div(len(matches), c_gt)),
            "mean_ned": statistics.fmean(m.ned for m in matches) if matches else 0.0,
            "mean_reward": statistics.fmean(m.reward for m in matches) if matches else 0.0,
        }
        total_images += len(per_image)
        total_gt += c_gt
        total_pred += c_pred
        print(
            f"{chunk}: images={len(per_image)} gt={c_gt} pred={c_pred} matched={len(matches)}"
        )

    # Aggregates
    num_matches = len(all_matches)
    det_p = safe_div(num_matches, total_pred)
    det_r = safe_div(num_matches, total_gt)
    exact_matches_ci = sum(m.exact_match_ci for m in all_matches)
    exact_matches = sum(m.exact_match for m in all_matches)
    e2e_p = safe_div(exact_matches, total_pred)
    e2e_r = safe_div(exact_matches, total_gt)

    rewards = [m.reward for m in all_matches]
    neds = [m.ned for m in all_matches]
    ious = [m.iou for m in all_matches]

    per_image_matched_means = [r["mean_reward_matched"] for r in all_per_image if r["num_matched"] > 0]
    per_image_over_gt = [r["reward_sum_over_gt"] for r in all_per_image if r["num_gt"] > 0]

    summary: dict[str, Any] = {
        "gt_parquet": str(args.gt_parquet),
        "results_root": str(args.results_root),
        "chunks": list(args.chunks),
        "iou_threshold": args.iou_threshold,
        "bridge_score_threshold": args.bridge_score_threshold,
        "counts": {
            "num_images": total_images,
            "num_images_with_gt": sum(1 for r in all_per_image if r["num_gt"] > 0),
            "num_images_with_pred": sum(1 for r in all_per_image if r["num_pred"] > 0),
            "total_gt": total_gt,
            "total_pred": total_pred,
            "pred_per_gt_ratio": safe_div(total_pred, total_gt),
            "gt_per_image": describe([r["num_gt"] for r in all_per_image]),
            "pred_per_image": describe([r["num_pred"] for r in all_per_image]),
            "pred_minus_gt_per_image": describe(
                [r["num_pred"] - r["num_gt"] for r in all_per_image]
            ),
        },
        "detection": {
            "num_matches": num_matches,
            "precision": det_p,
            "recall": det_r,
            "f1": f1(det_p, det_r),
            "mean_iou": statistics.fmean(ious) if ious else 0.0,
            "iou_histogram": iou_histogram(all_matches, args.iou_threshold),
        },
        "recognition_on_matched": {
            "exact_match_acc": safe_div(exact_matches, num_matches),
            "exact_match_acc_ci": safe_div(exact_matches_ci, num_matches),
            "mean_ned": statistics.fmean(neds) if neds else 0.0,
        },
        "end_to_end_exact": {
            "precision": e2e_p,
            "recall": e2e_r,
            "f1": f1(e2e_p, e2e_r),
        },
        "flow_grpo_reward": {
            "definition": "r_i = max(1 - lev(pred_rec_i, gt_text_i) / max(len(gt_text_i), 1), 0)",
            "mean_over_matched": statistics.fmean(rewards) if rewards else 0.0,
            "std_over_matched": statistics.pstdev(rewards) if len(rewards) > 1 else 0.0,
            "median_over_matched": statistics.median(rewards) if rewards else 0.0,
            "frac_perfect": safe_div(sum(1 for r in rewards if r > 0.9999), len(rewards)),
            "frac_zero": safe_div(sum(1 for r in rewards if r < 1e-9), len(rewards)),
            "mean_per_image_matched_only": (
                statistics.fmean(per_image_matched_means) if per_image_matched_means else 0.0
            ),
            "mean_per_image_over_gt": (
                statistics.fmean(per_image_over_gt) if per_image_over_gt else 0.0
            ),
            "histogram": reward_histogram(all_matches),
        },
        "per_chunk": per_chunk_summary,
    }

    # Write outputs
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (args.out_dir / "summary.md").write_text(render_markdown(summary), encoding="utf-8")
    write_csv(args.out_dir / "per_image.csv", all_per_image)
    write_csv(args.out_dir / "matched_pairs.csv", [asdict(m) for m in all_matches])
    write_csv(args.out_dir / "unmatched_gt.csv", [asdict(i) for i in all_unmatched_gt])
    write_csv(args.out_dir / "unmatched_pred.csv", [asdict(i) for i in all_unmatched_pred])

    print()
    print(f"Wrote summary to: {args.out_dir}/summary.md")
    print(f"   detection F1 = {summary['detection']['f1']*100:.2f}%   mean NED = {summary['recognition_on_matched']['mean_ned']:.4f}")
    print(f"   flow-grpo reward mean (over matched) = {summary['flow_grpo_reward']['mean_over_matched']:.4f}")


if __name__ == "__main__":
    main()
