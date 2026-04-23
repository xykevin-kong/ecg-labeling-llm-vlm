#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Step 3.1: Filter PTB-XL human-report labels by LLM consensus rules.

Input:
    {root_dir}/outputs/ptbxl/step2_ptbxl_human_report_filled.jsonl

Output:
    {root_dir}/outputs/ptbxl/step3_1_ptbxl_human_report_kept.jsonl
    {root_dir}/outputs/ptbxl/step3_1_ptbxl_human_report_dropped.jsonl
    {root_dir}/outputs/ptbxl/step3_1_ptbxl_human_report_filter_stats.json

Rules:
1. Keep directly if llm3.status is:
   - "pass"
   - "high_consistency"

2. If llm3.status == "conflict", keep only when:
   - llm3.final.report_label.mapped == llm1.report_label.mapped
   - OR llm3.final.report_label.mapped == llm2.report_label.mapped
   - OR llm3.final.report_label.mapped == union(llm1.report_label.mapped, llm2.report_label.mapped)

3. Otherwise drop.

Notes:
- Comparison is order-insensitive for mapped labels.
- The original full item is preserved in output JSONL.
- Additional helper fields are added:
    - keep_reason
    - step3_1_keep
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple


# =========================================================
# Argument parsing
# =========================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Filter PTB-XL step2 results by LLM consensus rules."
    )
    parser.add_argument(
        "--root_dir",
        type=str,
        required=True,
        help="Root directory of the ecg_preprocessing_pipeline project."
    )
    parser.add_argument(
        "--input_path",
        type=str,
        default=None,
        help="Optional custom input JSONL path. If not set, use default step2 output path."
    )
    parser.add_argument(
        "--output_keep_path",
        type=str,
        default=None,
        help="Optional custom kept JSONL output path."
    )
    parser.add_argument(
        "--output_drop_path",
        type=str,
        default=None,
        help="Optional custom dropped JSONL output path."
    )
    parser.add_argument(
        "--stats_path",
        type=str,
        default=None,
        help="Optional custom stats JSON path."
    )
    return parser.parse_args()


# =========================================================
# IO
# =========================================================
def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
    return data


def write_jsonl(path: Path, data_list: List[Dict[str, Any]]) -> None:
    ensure_parent_dir(path)
    with open(path, "w", encoding="utf-8") as f:
        for obj in data_list:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def write_json(path: Path, obj: Dict[str, Any]) -> None:
    ensure_parent_dir(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


# =========================================================
# Helpers
# =========================================================
def normalize_label_list(labels: List[Any]) -> List[str]:
    cleaned = []
    seen = set()

    for x in labels:
        s = str(x).strip()
        if not s:
            continue
        if s not in seen:
            seen.add(s)
            cleaned.append(s)

    return cleaned


def label_set(labels: List[Any]) -> set:
    return set(normalize_label_list(labels))


def get_mapped(obj: Dict[str, Any], path_type: str) -> List[str]:
    """
    path_type:
        - "llm1"
        - "llm2"
        - "llm3_final"
    """
    try:
        if path_type in ("llm1", "llm2"):
            labels = obj[path_type]["report_label"]["mapped"]
        elif path_type == "llm3_final":
            labels = obj["llm3"]["final"]["report_label"]["mapped"]
        else:
            return []
        return normalize_label_list(labels)
    except Exception:
        return []


def get_llm3_status(obj: Dict[str, Any]) -> str:
    try:
        return str(obj["llm3"]["status"]).strip()
    except Exception:
        return ""


def build_union(a: List[str], b: List[str]) -> List[str]:
    union_list = []
    seen = set()

    for x in a + b:
        if x not in seen:
            seen.add(x)
            union_list.append(x)

    return union_list


def mapped_equal(a: List[str], b: List[str]) -> bool:
    return label_set(a) == label_set(b)


# =========================================================
# Core filter logic
# =========================================================
def decide_keep(item: Dict[str, Any]) -> Tuple[bool, str]:
    status = get_llm3_status(item)

    if status == "pass":
        return True, "llm3_pass"

    if status == "high_consistency":
        return True, "llm3_high_consistency"

    if status != "conflict":
        return False, f"unexpected_status:{status}"

    llm1_mapped = get_mapped(item, "llm1")
    llm2_mapped = get_mapped(item, "llm2")
    llm3_mapped = get_mapped(item, "llm3_final")
    union_mapped = build_union(llm1_mapped, llm2_mapped)

    if mapped_equal(llm3_mapped, llm1_mapped):
        return True, "conflict_but_llm3_mapped_eq_llm1"

    if mapped_equal(llm3_mapped, llm2_mapped):
        return True, "conflict_but_llm3_mapped_eq_llm2"

    if mapped_equal(llm3_mapped, union_mapped):
        return True, "conflict_but_llm3_mapped_eq_union_llm1_llm2"

    return False, "conflict_not_matched"


# =========================================================
# Main
# =========================================================
def main() -> None:
    args = parse_args()
    root_dir = Path(args.root_dir).resolve()

    input_path = (
        Path(args.input_path).resolve()
        if args.input_path
        else root_dir / "outputs" / "ptbxl" / "step2_ptbxl_human_report_filled.jsonl"
    )

    output_keep_path = (
        Path(args.output_keep_path).resolve()
        if args.output_keep_path
        else root_dir / "outputs" / "ptbxl" / "step3_1_ptbxl_human_report_kept.jsonl"
    )

    output_drop_path = (
        Path(args.output_drop_path).resolve()
        if args.output_drop_path
        else root_dir / "outputs" / "ptbxl" / "step3_1_ptbxl_human_report_dropped.jsonl"
    )

    stats_path = (
        Path(args.stats_path).resolve()
        if args.stats_path
        else root_dir / "outputs" / "ptbxl" / "step3_1_ptbxl_human_report_filter_stats.json"
    )

    if not input_path.exists():
        raise FileNotFoundError(f"Input JSONL not found: {input_path}")

    data_list = load_jsonl(input_path)

    kept = []
    dropped = []

    stats = {
        "total": 0,
        "kept": 0,
        "dropped": 0,
        "keep_reason_counts": {},
        "drop_reason_counts": {},
        "status_counts": {}
    }

    for item in data_list:
        stats["total"] += 1

        status = get_llm3_status(item)
        stats["status_counts"][status] = stats["status_counts"].get(status, 0) + 1

        keep_flag, reason = decide_keep(item)

        out_item = dict(item)
        out_item["step3_1_keep"] = keep_flag
        out_item["keep_reason"] = reason

        if keep_flag:
            kept.append(out_item)
            stats["kept"] += 1
            stats["keep_reason_counts"][reason] = stats["keep_reason_counts"].get(reason, 0) + 1
        else:
            dropped.append(out_item)
            stats["dropped"] += 1
            stats["drop_reason_counts"][reason] = stats["drop_reason_counts"].get(reason, 0) + 1

    write_jsonl(output_keep_path, kept)
    write_jsonl(output_drop_path, dropped)
    write_json(stats_path, stats)

    print("\n================ STEP 3.1 SUMMARY ================")
    print(f"📄 Input total        : {stats['total']}")
    print(f"✅ Kept               : {stats['kept']}")
    print(f"❌ Dropped            : {stats['dropped']}")
    print(f"📁 Kept JSONL         : {output_keep_path}")
    print(f"📁 Dropped JSONL      : {output_drop_path}")
    print(f"📊 Stats JSON         : {stats_path}")

    print("\n[LLM3 status counts]")
    for k, v in sorted(stats["status_counts"].items(), key=lambda x: x[0]):
        print(f"  - {k}: {v}")

    print("\n[Keep reason counts]")
    for k, v in sorted(stats["keep_reason_counts"].items(), key=lambda x: x[0]):
        print(f"  - {k}: {v}")

    print("\n[Drop reason counts]")
    for k, v in sorted(stats["drop_reason_counts"].items(), key=lambda x: x[0]):
        print(f"  - {k}: {v}")

    print("\n✅ Step 3.1 finished.")


if __name__ == "__main__":
    main()