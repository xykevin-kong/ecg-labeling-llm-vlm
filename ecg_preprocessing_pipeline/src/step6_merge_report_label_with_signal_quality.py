#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Step 6: Merge Step3.4 final report labels with Step5 VLM signal quality results.

Input:
    {root_dir}/outputs/ptbxl/step3_4_ptxbl_report_llm_label.jsonl
    {root_dir}/outputs/ptbxl/step5_vlm_quality_results.jsonl

Output:
    {root_dir}/outputs/ptbxl/ptxbl_report_llm_label_with_vlm_quality.jsonl
    {root_dir}/outputs/ptbxl/step6_merge_report_label_with_vlm_quality_stats.jsonl

Rules:
1. Use Step3.4 report labels as the main table
2. Only merge Step5 entries when the two VLM results agree and both are 1
3. step5 ecg_id like "00006" should be converted to integer 6 before matching
4. For type == "drift", merge into "baseline_drift"
5. For type == "noise", merge into "noise"
6. Lead names are deduplicated by exact string match
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


# =========================================================
# Args
# =========================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="Merge step3.4 report labels with step5 VLM signal quality results."
    )
    parser.add_argument(
        "--root_dir",
        type=str,
        required=True,
        help="Root directory of ecg_preprocessing_pipeline."
    )
    parser.add_argument(
        "--report_label_path",
        type=str,
        default=None,
        help="Optional custom path for step3.4 output jsonl."
    )
    parser.add_argument(
        "--quality_path",
        type=str,
        default=None,
        help="Optional custom path for step5 success jsonl."
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Optional custom merged output path."
    )
    parser.add_argument(
        "--stats_path",
        type=str,
        default=None,
        help="Optional custom stats jsonl path."
    )
    return parser.parse_args()


# =========================================================
# IO
# =========================================================
def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def write_jsonl(path: Path, data: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def write_jsonl_single(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


# =========================================================
# Utils
# =========================================================
def dedup_keep_order(items: List[Any]) -> List[str]:
    seen = set()
    out = []
    for x in items:
        s = str(x).strip()
        if not s:
            continue
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out


def normalize_ecg_id(x: Any) -> int:
    """
    "00006" -> 6
    6 -> 6
    "6" -> 6
    """
    s = str(x).strip()
    return int(s)


def drift_agree_and_positive(item: Dict[str, Any]) -> bool:
    try:
        return (
            item["type"] == "drift"
            and item["vlm_result_1"]["baseline_drift"] == 1
            and item["vlm_result_2"]["baseline_drift"] == 1
        )
    except Exception:
        return False


def drift_disagree(item: Dict[str, Any]) -> bool:
    try:
        if item["type"] != "drift":
            return False
        v1 = item["vlm_result_1"]["baseline_drift"]
        v2 = item["vlm_result_2"]["baseline_drift"]
        return v1 != v2
    except Exception:
        return False


def noise_agree_and_positive(item: Dict[str, Any]) -> bool:
    try:
        return (
            item["type"] == "noise"
            and item["vlm_result_1"]["noise"] == 1
            and item["vlm_result_2"]["noise"] == 1
        )
    except Exception:
        return False


def noise_disagree(item: Dict[str, Any]) -> bool:
    try:
        if item["type"] != "noise":
            return False
        v1 = item["vlm_result_1"]["noise"]
        v2 = item["vlm_result_2"]["noise"]
        return v1 != v2
    except Exception:
        return False


# =========================================================
# Main
# =========================================================
def main():
    args = parse_args()
    root = Path(args.root_dir).resolve()

    report_label_path = (
        Path(args.report_label_path).resolve()
        if args.report_label_path
        else root / "outputs" / "ptbxl" / "step3_4_ptxbl_report_llm_label.jsonl"
    )

    quality_path = (
        Path(args.quality_path).resolve()
        if args.quality_path
        else root / "outputs" / "ptbxl" / "step5_vlm_quality_results.jsonl"
    )

    output_path = (
        Path(args.output_path).resolve()
        if args.output_path
        else root / "outputs" / "ptbxl" / "ptxbl_report_llm_label_with_vlm_quality.jsonl"
    )

    stats_path = (
        Path(args.stats_path).resolve()
        if args.stats_path
        else root / "outputs" / "ptbxl" / "step6_merge_report_label_with_quality_vlm_stats.jsonl"
    )

    if not report_label_path.exists():
        raise FileNotFoundError(f"Step3.4 report label file not found: {report_label_path}")
    if not quality_path.exists():
        raise FileNotFoundError(f"Step5 quality file not found: {quality_path}")

    report_data = load_jsonl(report_label_path)
    quality_data = load_jsonl(quality_path)

    stats = {
        "step3_4_total_samples": len(report_data),
        "step5_total_records": len(quality_data),

        "step5_drift_agree_positive": 0,
        "step5_noise_agree_positive": 0,
        "step5_drift_disagree": 0,
        "step5_noise_disagree": 0,

        "step5_agree_positive_total": 0,
        "step5_disagree_total": 0,

        "merged_drift_records": 0,
        "merged_noise_records": 0,

        "ecg_with_baseline_drift": 0,
        "ecg_with_noise": 0,
        "ecg_with_any_quality_flag": 0,
    }

    # 先以 step3.4 为主表建立索引
    merged_map: Dict[int, Dict[str, Any]] = {}
    for item in report_data:
        ecg_id = normalize_ecg_id(item["ecg_id"])
        merged_map[ecg_id] = {
            "ecg_id": ecg_id,
            "report_label": dedup_keep_order(item.get("report_label", []))
        }

    # 合并 step5
    for item in quality_data:
        try:
            ecg_id = normalize_ecg_id(item["ecg_id"])
            lead_name = str(item["lead_name"]).strip()
            if not lead_name:
                continue

            if drift_disagree(item):
                stats["step5_drift_disagree"] += 1
                stats["step5_disagree_total"] += 1
                continue

            if noise_disagree(item):
                stats["step5_noise_disagree"] += 1
                stats["step5_disagree_total"] += 1
                continue

            if ecg_id not in merged_map:
                continue

            row = merged_map[ecg_id]

            if drift_agree_and_positive(item):
                stats["step5_drift_agree_positive"] += 1
                stats["step5_agree_positive_total"] += 1

                old = row.get("baseline_drift", [])
                new = dedup_keep_order(old + [lead_name])
                row["baseline_drift"] = new
                stats["merged_drift_records"] += 1

            elif noise_agree_and_positive(item):
                stats["step5_noise_agree_positive"] += 1
                stats["step5_agree_positive_total"] += 1

                old = row.get("noise", [])
                new = dedup_keep_order(old + [lead_name])
                row["noise"] = new
                stats["merged_noise_records"] += 1

        except Exception:
            continue

    # 组装输出，并统计 ECG 级别数量
    output_rows = []
    for ecg_id in sorted(merged_map.keys()):
        row = merged_map[ecg_id]

        if "baseline_drift" in row:
            row["baseline_drift"] = dedup_keep_order(row["baseline_drift"])
            if row["baseline_drift"]:
                stats["ecg_with_baseline_drift"] += 1
            else:
                row.pop("baseline_drift", None)

        if "noise" in row:
            row["noise"] = dedup_keep_order(row["noise"])
            if row["noise"]:
                stats["ecg_with_noise"] += 1
            else:
                row.pop("noise", None)

        if ("baseline_drift" in row) or ("noise" in row):
            stats["ecg_with_any_quality_flag"] += 1

        output_rows.append(row)

    write_jsonl(output_path, output_rows)
    write_jsonl_single(stats_path, stats)

    print("\n================ STEP 6 SUMMARY ================")
    print(f"📄 Step3.4 total samples         : {stats['step3_4_total_samples']}")
    print(f"📄 Step5 total records           : {stats['step5_total_records']}")
    print(f"✅ Step5 drift agree+1          : {stats['step5_drift_agree_positive']}")
    print(f"✅ Step5 noise agree+1          : {stats['step5_noise_agree_positive']}")
    print(f"❌ Step5 drift disagree         : {stats['step5_drift_disagree']}")
    print(f"❌ Step5 noise disagree         : {stats['step5_noise_disagree']}")
    print(f"➕ Merged drift records         : {stats['merged_drift_records']}")
    print(f"➕ Merged noise records         : {stats['merged_noise_records']}")
    print(f"🩺 ECG with baseline_drift      : {stats['ecg_with_baseline_drift']}")
    print(f"🩺 ECG with noise               : {stats['ecg_with_noise']}")
    print(f"🩺 ECG with any quality flag    : {stats['ecg_with_any_quality_flag']}")
    print(f"📁 Output JSONL                 : {output_path}")
    print(f"📊 Stats JSONL                  : {stats_path}")
    print("✅ Step 6 finished.")


if __name__ == "__main__":
    main()