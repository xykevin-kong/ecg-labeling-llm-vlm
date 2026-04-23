#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Step 4: Compute lead-level ECG signal quality metrics on PTB-XL and select top 5% tails.

This script:
1. Loads PTB-XL metadata
2. Reads all ECG records from records500
3. Computes lead-level baseline drift and noise metrics
4. Estimates global distributions across all leads
5. Selects the top 5% tail for drift and noise
6. Writes a JSONL file for downstream VLM quality verification
7. Saves distribution plots for manuscript / preprint use

Usage:
    python src/step4_compute_signal_quality_top5.py \
        --root_dir /path/to/ecg_preprocessing_pipeline \
        --meta_path data/raw/ptbxl/ptbxl_database.csv \
        --max_workers 6
"""

from __future__ import annotations

import argparse
import json
import math
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import wfdb
from scipy.signal import butter, filtfilt
from tqdm import tqdm


# =========================================================
# Defaults
# =========================================================
DEFAULT_MAX_WORKERS = 6
DEFAULT_TOP_PERCENT = 5.0

STD_12_LEADS = [
    "I", "II", "III",
    "aVR", "aVL", "aVF",
    "V1", "V2", "V3", "V4", "V5", "V6"
]


# =========================================================
# Args
# =========================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute PTB-XL lead-level signal quality metrics and select top 5% tails."
    )
    parser.add_argument(
        "--root_dir",
        type=str,
        required=True,
        help="Root directory of ecg_preprocessing_pipeline."
    )
    parser.add_argument(
        "--meta_path",
        type=str,
        required=True,
        help="Path to metadata JSON, relative to root_dir or absolute path."
    )
    parser.add_argument(
        "--records_dir",
        type=str,
        default="data/raw/ptbxl/records500",
        help="Path to records500 directory, relative to root_dir or absolute path."
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=DEFAULT_MAX_WORKERS,
        help=f"Maximum worker threads (default: {DEFAULT_MAX_WORKERS})."
    )
    parser.add_argument(
        "--top_percent",
        type=float,
        default=DEFAULT_TOP_PERCENT,
        help=f"Top tail percentage for selection (default: {DEFAULT_TOP_PERCENT})."
    )
    parser.add_argument(
        "--plot_bins",
        type=int,
        default=120,
        help="Number of histogram bins for distribution plots."
    )
    return parser.parse_args()


# =========================================================
# I/O
# =========================================================
def resolve_path(root_dir: Path, path_str: str) -> Path:
    p = Path(path_str)
    if p.is_absolute():
        return p
    return root_dir / p


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# =========================================================
# Signal utilities
# =========================================================
def build_filters(fs: float):
    nyq = fs / 2.0

    b_lp, a_lp = butter(2, 0.5 / nyq, btype="low")

    hf_high = min(100.0, nyq - 1.0)
    if hf_high <= 20.0:
        hf_high = min(nyq * 0.95, 40.0)
    b_hf, a_hf = butter(2, [20.0 / nyq, hf_high / nyq], btype="band")

    b_ecg, a_ecg = butter(2, [0.5 / nyq, 40.0 / nyq], btype="band")
    return (b_lp, a_lp), (b_hf, a_hf), (b_ecg, a_ecg)


def apply_filter(sig: np.ndarray, coeff) -> np.ndarray:
    b, a = coeff
    return filtfilt(b, a, sig)


def longest_run_bool(arr_bool: np.ndarray) -> int:
    max_run = 0
    cur = 0
    for x in arr_bool:
        if x:
            cur += 1
            max_run = max(max_run, cur)
        else:
            cur = 0
    return max_run


def sign_flip_ratio(x: np.ndarray) -> float:
    d = np.diff(x)
    if len(d) < 3:
        return 0.0
    s = np.sign(d)
    s = s[s != 0]
    if len(s) < 3:
        return 0.0
    return float(np.mean(s[1:] * s[:-1] < 0))


def zero_crossing_rate(x: np.ndarray) -> float:
    x = x - np.median(x)
    s = np.sign(x)
    s = s[s != 0]
    if len(s) < 3:
        return 0.0
    return float(np.mean(s[1:] * s[:-1] < 0))


# =========================================================
# Core metric computation
# =========================================================
def compute_lead_metrics(sig: np.ndarray, fs: float, lp_f, hf_f, ecg_f) -> Tuple[float, float]:
    # ----- baseline drift -----
    base = apply_filter(sig, lp_f)
    drift = float(np.std(base))

    # ----- noise -----
    abs_sig = np.abs(sig)
    amp_thr = np.percentile(abs_sig, 98)

    diff_sig = np.abs(np.diff(sig))
    diff_thr = np.percentile(diff_sig, 98)

    slope_mask = np.concatenate([[True], diff_sig < diff_thr])
    amp_mask = abs_sig < amp_thr
    mask = amp_mask & slope_mask

    hf_full = apply_filter(sig, hf_f)
    ecg_full = apply_filter(sig, ecg_f)

    hf_masked = hf_full[mask]

    if len(hf_masked) < 50:
        return drift, 0.0

    base_noise = float(np.std(hf_masked) / (np.std(ecg_full) + 1e-6))

    win_size = int(fs * 0.5)
    step = int(fs * 0.25)

    win_scores = []

    for start in range(0, len(sig) - win_size, step):
        sub_mask = mask[start:start + win_size]
        if np.sum(sub_mask) < win_size * 0.6:
            continue

        hf_win = hf_full[start:start + win_size][sub_mask]
        ecg_win = ecg_full[start:start + win_size][sub_mask]
        raw_win = sig[start:start + win_size][sub_mask]

        if len(hf_win) < 20:
            continue

        local_hf_ratio = float(np.std(hf_win) / (np.std(ecg_win) + 1e-6))
        flip_ratio = sign_flip_ratio(raw_win)
        zc_ratio = zero_crossing_rate(hf_win)

        score = local_hf_ratio * (1.0 + 2.0 * flip_ratio) * (1.0 + 1.5 * zc_ratio)
        win_scores.append(score)

    if len(win_scores) == 0:
        return drift, 0.0

    win_scores = np.asarray(win_scores)
    local_thr = np.percentile(win_scores, 90)

    high_flags = win_scores > local_thr
    high_ratio = float(np.mean(high_flags))
    max_consecutive = longest_run_bool(high_flags)

    noise = base_noise * (1.0 + 2.0 * high_ratio) * (1.0 + max_consecutive / 4.0)

    if base_noise < 0.08:
        noise = 0.0

    return drift, float(noise)


def process_one(item: Dict[str, Any], records_dir: Path) -> Optional[List[Dict[str, Any]]]:
    ecg_id = item["ecg_id"]
    filename_hr = item["filename_hr"]

    # filename_hr 例如: records500/00000/00001_hr
    # 这里我们只需要拼到 root 下的 records_dir
    # 所以取 filename_hr 相对 records500 后面的部分
    parts = Path(filename_hr).parts
    try:
        idx = parts.index("records500")
        rel_after_records500 = Path(*parts[idx + 1:])
    except ValueError:
        rel_after_records500 = Path(filename_hr)

    record_path = records_dir / rel_after_records500

    try:
        record = wfdb.rdrecord(str(record_path))
        sigs = record.p_signal
        fs = float(record.fs)

        lp_f, hf_f, ecg_f = build_filters(fs)

        n_leads = min(12, sigs.shape[1])
        result = []

        for i in range(n_leads):
            sig = sigs[:, i]
            drift, noise = compute_lead_metrics(sig, fs, lp_f, hf_f, ecg_f)

            result.append({
                "ecg_id": ecg_id,
                "lead_idx": i,
                "lead_name": STD_12_LEADS[i],
                "filename_hr": filename_hr,
                "record_path": str(record_path),
                "drift": float(drift),
                "noise": float(noise),
            })

        return result

    except Exception as e:
        print(f"❌ Error ecg_id={ecg_id}: {e}")
        return None


# =========================================================
# Plotting
# =========================================================
def save_distribution_plot(
    values: np.ndarray,
    threshold: float,
    title: str,
    xlabel: str,
    save_path: Path,
    bins: int,
):
    fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=400)  # 🔥 300 → 400（更高清）

    # ===== 背景 =====
    fig.patch.set_facecolor("#f7f7f5")
    ax.set_facecolor("#f7f7f5")

    # ===== 数据 =====
    values_clip = values[values <= 2.0]

    # ===== histogram =====
    ax.hist(
        values_clip,
        bins=np.linspace(0, 2, 70),
        color="#4a6cf7",
        edgecolor="#ffffff",
        linewidth=0.5,   # 🔥 0.6 → 0.5（更细更锐）
        alpha=0.9        # 🔥 稍微提高一点清晰度
    )

    # ===== 阈值线 =====
    ax.axvline(
        threshold,
        color="#e85d75",
        linestyle=(0, (4, 4)),
        linewidth=1.2,   # 🔥 1.6 → 1.2（更精细）
        label=f"Top 5% threshold = {threshold:.2f}"
    )

    # ===== X轴 =====
    ax.set_xlim(0, 1.0)
    ax.set_xticks([0, 0.5, 1.0])

    # ===== 标签 =====
    ax.set_xlabel(xlabel, fontsize=11, color="#222222")
    ax.set_ylabel("Count", fontsize=11, color="#222222")
    ax.set_title(title, fontsize=15, color="#111111", pad=14)

    # ===== 网格 =====
    ax.grid(axis="y", color="#000000", alpha=0.06, linewidth=0.6)  # 🔥 更细
    ax.grid(False, axis="x")

    # ===== 边框 =====
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.spines["left"].set_color("black")
    ax.spines["bottom"].set_color("black")

    ax.spines["left"].set_linewidth(0.8)   # 🔥 更细更锐
    ax.spines["bottom"].set_linewidth(0.8)

    # ===== 刻度 =====
    ax.tick_params(axis="both", colors="#222222", labelsize=10, width=0.8)

    # ===== legend =====
    ax.legend(frameon=False, fontsize=9, loc="upper right")

    # 🔥 关键：避免边缘模糊
    plt.tight_layout()
    plt.savefig(save_path, dpi=400, bbox_inches="tight")  # 🔥 高清输出
    plt.close()

# =========================================================
# Main
# =========================================================
def main() -> None:
    args = parse_args()

    root_dir = Path(args.root_dir).resolve()
    meta_path = resolve_path(root_dir, args.meta_path)
    records_dir = resolve_path(root_dir, args.records_dir)

    output_dir = root_dir / "outputs" / "ptbxl"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_jsonl = output_dir / "step4_signal_quality_top5.jsonl"
    output_summary = output_dir / "step4_signal_quality_summary.json"
    drift_plot_path = output_dir / "step4_drift_distribution.png"
    noise_plot_path = output_dir / "step4_noise_distribution.png"

    if not meta_path.exists():
        raise FileNotFoundError(f"Metadata JSON not found: {meta_path}")
    if not records_dir.exists():
        raise FileNotFoundError(f"records500 directory not found: {records_dir}")

    df = pd.read_csv(meta_path)
    data = []
    for i, row in df.iterrows():
        if pd.isna(row["filename_hr"]):
            continue

        data.append({
            "ecg_id": int(row["ecg_id"]),
            "filename_hr": str(row["filename_hr"])
        })
    if not isinstance(data, list):
        raise ValueError("Metadata JSON must be a list of dict items.")

    print(f"📄 Total ECG samples: {len(data)}")
    print(f"📂 records_dir: {records_dir}")
    print(f"📂 output_dir : {output_dir}")

    all_leads: List[Dict[str, Any]] = []

    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = [executor.submit(process_one, item, records_dir) for item in data]

        for future in tqdm(as_completed(futures), total=len(futures)):
            res = future.result()
            if res:
                all_leads.extend(res)

    if len(all_leads) == 0:
        raise RuntimeError("No valid lead results were produced.")

    print(f"✅ Total processed leads: {len(all_leads)}")

    all_drift = np.asarray([x["drift"] for x in all_leads], dtype=float)
    all_noise = np.asarray([x["noise"] for x in all_leads], dtype=float)

    tail_percentile = 100.0 - args.top_percent
    drift_thr = float(np.percentile(all_drift, tail_percentile))
    noise_thr = float(np.percentile(all_noise, tail_percentile))

    print("\n==== DRIFT PERCENTILES ====")
    for p in [90, 95, 97, 98, 99]:
        print(f"{p}: {np.percentile(all_drift, p):.6f}")

    print("\n==== NOISE PERCENTILES ====")
    for p in [90, 95, 97, 98, 99]:
        print(f"{p}: {np.percentile(all_noise, p):.6f}")

    print(f"\n✅ Selected tail percentage: top {args.top_percent:.1f}%")
    print(f"✅ Drift threshold (p{tail_percentile:.1f}): {drift_thr:.6f}")
    print(f"✅ Noise threshold (p{tail_percentile:.1f}): {noise_thr:.6f}")

    # Write JSONL
    n_drift = 0
    n_noise = 0
    with open(output_jsonl, "w", encoding="utf-8") as f:
        for x in all_leads:
            if x["drift"] >= drift_thr:
                row = {
                    "ecg_id": x["ecg_id"],
                    "lead_name": x["lead_name"],
                    "type": "drift",
                    "percentile": f"top_{int(args.top_percent)}%",
                    "value": x["drift"]
                }
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                n_drift += 1

            if x["noise"] >= noise_thr:
                row = {
                    "ecg_id": x["ecg_id"],
                    "lead_name": x["lead_name"],
                    "type": "noise",
                    "percentile": f"top_{int(args.top_percent)}%",
                    "value": x["noise"]
                }
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                n_noise += 1

    # Save summary
    summary = {
        "total_ecg_samples": len(data),
        "total_processed_leads": len(all_leads),
        "top_percent": args.top_percent,
        "drift_threshold": drift_thr,
        "noise_threshold": noise_thr,
        "drift_selected_count": n_drift,
        "noise_selected_count": n_noise,
        "drift_percentiles": {
            "90": float(np.percentile(all_drift, 90)),
            "95": float(np.percentile(all_drift, 95)),
            "97": float(np.percentile(all_drift, 97)),
            "98": float(np.percentile(all_drift, 98)),
            "99": float(np.percentile(all_drift, 99)),
        },
        "noise_percentiles": {
            "90": float(np.percentile(all_noise, 90)),
            "95": float(np.percentile(all_noise, 95)),
            "97": float(np.percentile(all_noise, 97)),
            "98": float(np.percentile(all_noise, 98)),
            "99": float(np.percentile(all_noise, 99)),
        },
        "output_jsonl": str(output_jsonl),
        "drift_plot": str(drift_plot_path),
        "noise_plot": str(noise_plot_path),
    }

    with open(output_summary, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # Save plots
    save_distribution_plot(
        values=all_drift,
        threshold=drift_thr,
        title=f"Lead-level baseline drift distribution",
        xlabel="Baseline drift score",
        save_path=drift_plot_path,
        bins=args.plot_bins,
    )

    save_distribution_plot(
        values=all_noise,
        threshold=noise_thr,
        title=f"Lead-level noise score distribution",
        xlabel="Noise score",
        save_path=noise_plot_path,
        bins=args.plot_bins,
    )

    print("\n================ FINAL SUMMARY ================")
    print(f"✅ JSONL saved : {output_jsonl}")
    print(f"✅ Summary saved: {output_summary}")
    print(f"✅ Drift plot   : {drift_plot_path}")
    print(f"✅ Noise plot   : {noise_plot_path}")
    print("✅ Step 4 finished.")


if __name__ == "__main__":
    main()