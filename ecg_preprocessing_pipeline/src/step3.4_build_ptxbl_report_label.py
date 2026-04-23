#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Step 3.4: Merge step3.1 mapped labels with step3.3 unmapped-term mappings,
then apply rule-based pruning to generate final PTB-XL report labels.

Output:
    step3_4_ptxbl_report_llm_label.jsonl
    step3_4_ptbxl_report_label_stats.jsonl
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", type=str, required=True)
    return parser.parse_args()


# =========================================================
# IO
# =========================================================
def load_jsonl(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def write_jsonl(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for x in data:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")


def write_jsonl_single(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


# =========================================================
# Utils
# =========================================================
def dedup(items):
    seen = set()
    out = []
    for x in items:
        x = str(x).strip()
        if x and x not in seen:
            seen.add(x)
            out.append(x)
    return out


# =========================================================
# Extract
# =========================================================
def get_mapped(item):
    try:
        return dedup(item["llm3"]["final"]["report_label"]["mapped"])
    except:
        return []


def get_unmapped(item):
    try:
        return dedup(item["llm3"]["final"]["report_label"]["unmapped"])
    except:
        return []


def build_term_map(step3_3):
    m = {}
    for x in step3_3:
        term = str(x.get("input", "")).strip()
        mapped = dedup(x.get("llm3", {}).get("mapped", []))
        if term:
            m[term] = mapped
    return m


# =========================================================
# Rules
# =========================================================
NORMAL = "正常心电图"
SINUS = "窦性心律"
CONFLICT = {"心房扑动", "心室颤动", "心房颤动", "心室扑动"}


def apply_rules(labels, stats):
    labels = dedup(labels)

    # Rule 3
    if SINUS in labels and any(x in labels for x in CONFLICT):
        labels = [x for x in labels if x != SINUS]
        stats["rule3_delete_sinus"] += 1

    # Rule 1
    if NORMAL in labels:
        s = set(labels)
        if s != {SINUS, NORMAL} and s != {NORMAL}:
            labels = [x for x in labels if x != NORMAL]
            stats["rule1_delete_normal"] += 1

    # Rule 2
    if labels == [SINUS]:
        labels = [SINUS, NORMAL]
        stats["rule2_add_normal"] += 1

    return dedup(labels)


# =========================================================
# Main
# =========================================================
def main():
    args = parse_args()
    root = Path(args.root_dir)

    step3_1_path = root / "outputs/ptbxl/step3_1_ptbxl_human_report_kept.jsonl"
    step3_3_path = root / "outputs/ptbxl/step3_3_ptbxl_consensus_unmapped_terms_filled.jsonl"

    output_path = root / "outputs/ptbxl/step3_4_ptxbl_report_llm_label.jsonl"
    stats_path = root / "outputs/ptbxl/step3_4_ptbxl_report_label_stats.jsonl"

    data1 = load_jsonl(step3_1_path)
    data3 = load_jsonl(step3_3_path)

    term_map = build_term_map(data3)

    stats = {
        "total": len(data1),
        "unmapped_samples": 0,
        "hit_samples": 0,
        "added_samples": 0,
        "total_terms": 0,
        "hit_terms": 0,
        "added_labels": 0,
        "rule1_delete_normal": 0,
        "rule2_add_normal": 0,
        "rule3_delete_sinus": 0,
    }

    out = []

    for item in data1:
        ecg_id = item["ecg_id"]

        mapped = get_mapped(item)
        unmapped = get_unmapped(item)

        if unmapped:
            stats["unmapped_samples"] += 1

        merged = list(mapped)
        added_flag = False
        hit_flag = False

        for t in unmapped:
            stats["total_terms"] += 1

            m = term_map.get(t, [])
            if m:
                stats["hit_terms"] += 1
                hit_flag = True

                before = len(merged)
                merged.extend(m)
                merged = dedup(merged)

                if len(merged) > before:
                    stats["added_labels"] += len(merged) - before
                    added_flag = True

        if hit_flag:
            stats["hit_samples"] += 1

        if added_flag:
            stats["added_samples"] += 1

        merged = apply_rules(merged, stats)

        out.append({
            "ecg_id": ecg_id,
            "report_label": merged
        })

    write_jsonl(output_path, out)
    write_jsonl_single(stats_path, stats)

    print("\n===== STEP 3.4 DONE =====")
    print("output:", output_path)
    print("stats :", stats_path)


if __name__ == "__main__":
    main()