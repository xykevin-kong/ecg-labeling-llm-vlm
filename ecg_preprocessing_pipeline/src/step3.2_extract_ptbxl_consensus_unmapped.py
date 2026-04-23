#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Step 3.2: Extract consensus-supported unmapped terms from kept PTB-XL samples.

Input:
    {root_dir}/outputs/ptbxl/step3_1_ptbxl_human_report_kept.jsonl

Output:
    {root_dir}/outputs/ptbxl/step3_2_ptbxl_consensus_unmapped.jsonl
    {root_dir}/outputs/ptbxl/step3_2_ptbxl_consensus_unmapped_dropped.jsonl
    {root_dir}/outputs/ptbxl/step3_2_ptbxl_consensus_unmapped_stats.json

Rules:
1. Only process samples already kept by Step 3.1.
2. Only keep samples whose final unmapped is non-empty.
3. Keep unmapped only when one of the following holds:
   - llm1.unmapped == llm2.unmapped
   - llm3.final.unmapped == llm1.unmapped
   - llm3.final.unmapped == llm2.unmapped
   - llm3.final.unmapped == union(llm1.unmapped, llm2.unmapped)

Otherwise, do not include that sample in the final unmapped file.

Final output JSONL format:
    {"ecg_id": 123, "unmapped": [...]}

Notes:
- Comparison is order-insensitive.
- Output only contains:
    - ecg_id
    - unmapped
- Empty unmapped will not be exported into the final kept output.
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
        description="Extract consensus-supported unmapped terms from Step 3.1 kept PTB-XL samples."
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
        help="Optional custom input JSONL path. If not set, use default Step 3.1 kept path."
    )
    parser.add_argument(
        "--output_keep_path",
        type=str,
        default=None,
        help="Optional custom consensus-unmapped JSONL output path."
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
        help="Optional custom stats JSON output path."
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
def normalize_list(items: List[Any]) -> List[str]:
    cleaned = []
    seen = set()

    for x in items:
        s = str(x).strip()
        if not s:
            continue
        if s not in seen:
            seen.add(s)
            cleaned.append(s)

    return cleaned


def as_set(items: List[Any]) -> set:
    return set(normalize_list(items))


def equal_ignore_order(a: List[Any], b: List[Any]) -> bool:
    return as_set(a) == as_set(b)


def build_union(a: List[Any], b: List[Any]) -> List[str]:
    result = []
    seen = set()

    for x in normalize_list(a) + normalize_list(b):
        if x not in seen:
            seen.add(x)
            result.append(x)

    return result


def get_unmapped(obj: Dict[str, Any], source: str) -> List[str]:
    """
    source:
        - llm1
        - llm2
        - llm3_final
    """
    try:
        if source in ("llm1", "llm2"):
            items = obj[source]["report_label"]["unmapped"]
        elif source == "llm3_final":
            items = obj["llm3"]["final"]["report_label"]["unmapped"]
        else:
            return []
        return normalize_list(items)
    except Exception:
        return []


# =========================================================
# Core logic
# =========================================================
def decide_unmapped_keep(item: Dict[str, Any]) -> Tuple[bool, str, List[str]]:
    llm1_unmapped = get_unmapped(item, "llm1")
    llm2_unmapped = get_unmapped(item, "llm2")
    llm3_unmapped = get_unmapped(item, "llm3_final")

    # final unmapped 为空，直接不输出
    if len(llm3_unmapped) == 0:
        return False, "final_unmapped_empty", []

    if equal_ignore_order(llm1_unmapped, llm2_unmapped):
        return True, "llm1_eq_llm2", llm1_unmapped

    if equal_ignore_order(llm3_unmapped, llm1_unmapped):
        return True, "llm3_eq_llm1", llm3_unmapped

    if equal_ignore_order(llm3_unmapped, llm2_unmapped):
        return True, "llm3_eq_llm2", llm3_unmapped

    union_unmapped = build_union(llm1_unmapped, llm2_unmapped)
    if equal_ignore_order(llm3_unmapped, union_unmapped):
        return True, "llm3_eq_union_llm1_llm2", llm3_unmapped

    return False, "unmapped_conflict_not_matched", []


# =========================================================
# Main
# =========================================================
def main() -> None:
    args = parse_args()
    root_dir = Path(args.root_dir).resolve()

    input_path = (
        Path(args.input_path).resolve()
        if args.input_path
        else root_dir / "outputs" / "ptbxl" / "step3_1_ptbxl_human_report_kept.jsonl"
    )

    output_keep_path = (
        Path(args.output_keep_path).resolve()
        if args.output_keep_path
        else root_dir / "outputs" / "ptbxl" / "step3_2_ptbxl_consensus_unmapped.jsonl"
    )

    output_drop_path = (
        Path(args.output_drop_path).resolve()
        if args.output_drop_path
        else root_dir / "outputs" / "ptbxl" / "step3_2_ptbxl_consensus_unmapped_dropped.jsonl"
    )

    stats_path = (
        Path(args.stats_path).resolve()
        if args.stats_path
        else root_dir / "outputs" / "ptbxl" / "step3_2_ptbxl_consensus_unmapped_stats.json"
    )

    if not input_path.exists():
        raise FileNotFoundError(f"Input JSONL not found: {input_path}")

    data_list = load_jsonl(input_path)

    kept = []
    dropped = []

    stats = {
        "total_kept_input_samples": 0,
        "consensus_unmapped_kept": 0,
        "consensus_unmapped_dropped": 0,
        "keep_reason_counts": {},
        "drop_reason_counts": {}
    }

    for item in data_list:
        stats["total_kept_input_samples"] += 1

        keep_flag, reason, final_unmapped = decide_unmapped_keep(item)

        if keep_flag:
            kept.append({
                "ecg_id": item["ecg_id"],
                "unmapped": final_unmapped
            })
            stats["consensus_unmapped_kept"] += 1
            stats["keep_reason_counts"][reason] = stats["keep_reason_counts"].get(reason, 0) + 1
        else:
            dropped.append({
                "ecg_id": item["ecg_id"],
                "unmapped": get_unmapped(item, "llm3_final"),
                "drop_reason": reason
            })
            stats["consensus_unmapped_dropped"] += 1
            stats["drop_reason_counts"][reason] = stats["drop_reason_counts"].get(reason, 0) + 1

    write_jsonl(output_keep_path, kept)
    write_jsonl(output_drop_path, dropped)
    write_json(stats_path, stats)

    print("\n================ STEP 3.2 SUMMARY ================")
    print(f"📄 Step3.1 kept input total         : {stats['total_kept_input_samples']}")
    print(f"✅ Consensus unmapped kept          : {stats['consensus_unmapped_kept']}")
    print(f"❌ Consensus unmapped dropped       : {stats['consensus_unmapped_dropped']}")
    print(f"📁 Consensus unmapped JSONL         : {output_keep_path}")
    print(f"📁 Dropped unmapped JSONL           : {output_drop_path}")
    print(f"📊 Stats JSON                       : {stats_path}")

    print("\n[Keep reason counts]")
    for k, v in sorted(stats["keep_reason_counts"].items(), key=lambda x: x[0]):
        print(f"  - {k}: {v}")

    print("\n[Drop reason counts]")
    for k, v in sorted(stats["drop_reason_counts"].items(), key=lambda x: x[0]):
        print(f"  - {k}: {v}")

    print("\n✅ Step 3.2 finished.")


if __name__ == "__main__":
    main()