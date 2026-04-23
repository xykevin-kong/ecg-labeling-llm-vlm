#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Step 5: VLM-based signal quality verification for candidate noisy / drift leads.

This script:
1. Loads candidate lead-level signal quality samples from Step 4 JSONL
2. Resolves ECG record paths from PTB-XL metadata
3. Renders single-lead ECG images
4. Queries a VLM twice independently for each candidate
5. Saves valid dual-VLM results for downstream consistency filtering
6. Writes failed samples to a separate error JSONL

Usage:
    export DASHSCOPE_API_KEY="your_api_key"

    python src/step5_vlm_signal_quality_verification.py \
        --root_dir /path/to/ecg_preprocessing_pipeline \
        --meta_path data/raw/ptbxl/ptbxl_database.csv \
        --input_jsonl outputs/ptbxl/step4_signal_quality_top5.jsonl \
        --records_dir data/raw/ptbxl/records500 \
        --max_workers 3

Notes:
- Resume is supported: completed (ecg_id, lead_name, type) tuples in success output will be skipped.
- Each sample is queried twice with the same VLM for internal agreement checking.
- This script only performs signal quality verification, not disease diagnosis.
- Invalid / failed outputs are written to a separate error JSONL and do NOT pollute the main result file.
"""

from __future__ import annotations

import argparse
import base64
import csv
import json
import os
import re
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import wfdb
from openai import OpenAI
from tqdm import tqdm


# =========================================================
# Defaults
# =========================================================
DEFAULT_MAX_WORKERS = 3
DEFAULT_MAX_RETRY = 3
MODEL_NAME_DEFAULT = "qwen3-vl-flash"

STD_12_LEADS = [
    "I", "II", "III",
    "aVR", "aVL", "aVF",
    "V1", "V2", "V3", "V4", "V5", "V6"
]

write_lock = threading.Lock()


# =========================================================
# Args
# =========================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="VLM-based signal quality verification for PTB-XL candidate leads."
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
        help="Path to PTB-XL metadata CSV, relative to root_dir or absolute path."
    )
    parser.add_argument(
        "--input_jsonl",
        type=str,
        required=True,
        help="Path to candidate JSONL from Step 4, relative to root_dir or absolute path."
    )
    parser.add_argument(
        "--records_dir",
        type=str,
        default="data/raw/ptbxl/records500",
        help="Path to records500 directory, relative to root_dir or absolute path."
    )
    parser.add_argument(
        "--output_jsonl",
        type=str,
        default="outputs/ptbxl/step5_vlm_quality_results.jsonl",
        help="Success output JSONL path, relative to root_dir or absolute path."
    )
    parser.add_argument(
        "--error_jsonl",
        type=str,
        default="outputs/ptbxl/step5_vlm_quality_error.jsonl",
        help="Error JSONL path, relative to root_dir or absolute path."
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=DEFAULT_MAX_WORKERS,
        help=f"Maximum worker threads (default: {DEFAULT_MAX_WORKERS})."
    )
    parser.add_argument(
        "--max_retry",
        type=int,
        default=DEFAULT_MAX_RETRY,
        help=f"Maximum retry count for each VLM call (default: {DEFAULT_MAX_RETRY})."
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default=MODEL_NAME_DEFAULT,
        help=f"VLM model name (default: {MODEL_NAME_DEFAULT})."
    )
    parser.add_argument(
        "--base_url",
        type=str,
        default="https://dashscope.aliyuncs.com/compatible-mode/v1",
        help="Compatible OpenAI-style API base URL."
    )
    return parser.parse_args()


# =========================================================
# Path / IO
# =========================================================
def resolve_path(root_dir: Path, path_str: str) -> Path:
    p = Path(path_str)
    if p.is_absolute():
        return p
    return root_dir / p


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


# =========================================================
# Metadata
# =========================================================
def load_id_to_path_from_csv(meta_path: Path) -> Dict[str, str]:
    id_to_path: Dict[str, str] = {}
    with open(meta_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                ecg_id = str(int(float(row["ecg_id"]))).zfill(5)
                filename_hr = row["filename_hr"]
                if filename_hr:
                    id_to_path[ecg_id] = filename_hr
            except Exception:
                continue
    return id_to_path


# =========================================================
# Resume
# =========================================================
def is_valid_success_item(item: Dict[str, Any]) -> bool:
    return (
        isinstance(item, dict)
        and "ecg_id" in item
        and "lead_name" in item
        and "type" in item
        and "vlm_result_1" in item
        and "vlm_result_2" in item
    )


def load_done_set(output_jsonl: Path) -> set:
    done_set = set()
    if not output_jsonl.exists():
        return done_set

    with open(output_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            try:
                item = json.loads(line)
                if not is_valid_success_item(item):
                    continue

                ecg_id = str(item["ecg_id"]).zfill(5)
                key = (ecg_id, item["lead_name"], item["type"])
                done_set.add(key)
            except Exception:
                continue

    return done_set


# =========================================================
# ECG rendering
# =========================================================
def draw_ecg(sig: np.ndarray, save_path: Path, fs: float) -> None:
    height = 2.5
    width = height * 6.25

    fig = plt.figure(figsize=(width, height), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])

    t = np.arange(len(sig)) / fs
    ax.plot(t, sig, color="black", linewidth=1)

    ax.set_xlim(0, 10)
    ax.set_ylim(-2, 2)
    ax.set_xticks([])
    ax.set_yticks([])

    for spine in ax.spines.values():
        spine.set_visible(False)

    plt.savefig(save_path, dpi=100, bbox_inches=None, pad_inches=0)
    plt.close(fig)


# =========================================================
# Prompt
# =========================================================
DRIFT_PROMPT = """
你是心电图信号质量评估专家。

任务：只判断是否存在“基线漂移”，禁止做任何疾病诊断。

判断标准（非常重要）：
- 正常：基线大致在同一水平线上（稳定）
- 漂移：整条信号的基线整体上下移动，不在同一水平线上（缓慢漂移）
- 忽略：P波、QRS波、T波及其形态变化

请根据“基线是否整体不在同一水平线”进行判断。

输出要求（严格遵守）：
1. 只能输出一行 JSON
2. 不能包含任何解释、文字或 ``` 符号
3. key 必须是 baseline_drift
4. value 只能是数字 0 或 1

正确格式示例：
{"baseline_drift":1}
"""

NOISE_PROMPT = """
你是心电图信号质量评估专家。

任务：只判断是否存在“干扰（噪声）”，禁止做任何疾病诊断。

判断标准（非常重要）：
- 噪声：出现快速、细碎、不规则的抖动（类似“毛刺”“发毛”“锯齿”）
- 必须是持续存在的随机抖动，而不是单个尖峰
- 忽略：P波、QRS波、T波及其病理变化

请根据“是否存在明显的非生理高频抖动”进行判断。

输出要求（严格遵守）：
1. 只能输出一行 JSON
2. 不能包含任何解释、文字或 ``` 符号
3. key 必须是 noise
4. value 只能是数字 0 或 1

正确格式示例：
{"noise":1}
"""


# =========================================================
# VLM utils
# =========================================================
def encode_image(image_path: Path) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def extract_json(text: str) -> Optional[Dict[str, Any]]:
    try:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            return json.loads(match.group())
    except Exception:
        pass
    return None


def validate_result(res: Dict[str, Any], sample_type: str) -> bool:
    if not isinstance(res, dict):
        return False

    if sample_type == "drift":
        return res.get("baseline_drift") in [0, 1]

    if sample_type == "noise":
        return res.get("noise") in [0, 1]

    return False


def call_vlm_with_retry(
    client: OpenAI,
    model_name: str,
    image_path: Path,
    prompt: str,
    sample_type: str,
    max_retry: int = 3,
) -> Dict[str, Any]:
    last_error = None

    for _ in range(max_retry):
        try:
            time.sleep(0.1)
            base64_img = encode_image(image_path)

            completion = client.chat.completions.create(
                model=model_name,
                messages=[{
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64_img}"
                            },
                        },
                        {
                            "type": "text",
                            "text": prompt
                        },
                    ],
                }],
            )

            text = completion.choices[0].message.content
            res = extract_json(text)

            if validate_result(res, sample_type):
                return {
                    "ok": True,
                    "result": res
                }

            last_error = {
                "stage": "invalid_json_or_schema",
                "raw_response": text
            }

        except Exception as e:
            last_error = {
                "stage": "api_exception",
                "error": str(e)
            }

    return {
        "ok": False,
        "error": last_error if last_error is not None else {"stage": "unknown_failure"}
    }


# =========================================================
# Single sample
# =========================================================
def process_one(
    item: Dict[str, Any],
    id_to_path: Dict[str, str],
    records_dir: Path,
    client: OpenAI,
    model_name: str,
    max_retry: int,
) -> Dict[str, Any]:
    ecg_id = str(item["ecg_id"]).zfill(5)
    lead_name = item["lead_name"]
    sample_type = item["type"]

    if ecg_id not in id_to_path:
        return {
            "success": False,
            "payload": {
                "ecg_id": ecg_id,
                "lead_name": lead_name,
                "type": sample_type,
                "error_stage": "metadata_lookup",
                "error": "ecg_id_not_found_in_metadata"
            }
        }

    if lead_name not in STD_12_LEADS:
        return {
            "success": False,
            "payload": {
                "ecg_id": ecg_id,
                "lead_name": lead_name,
                "type": sample_type,
                "error_stage": "lead_lookup",
                "error": "invalid_lead_name"
            }
        }

    filename_hr = id_to_path[ecg_id]
    parts = Path(filename_hr).parts

    try:
        idx = parts.index("records500")
        rel_after_records500 = Path(*parts[idx + 1:])
    except ValueError:
        rel_after_records500 = Path(filename_hr)

    record_path = records_dir / rel_after_records500
    img_path = Path("/tmp") / f"{uuid.uuid4().hex}.png"

    try:
        record = wfdb.rdrecord(str(record_path))
        lead_idx = STD_12_LEADS.index(lead_name)
        sig = record.p_signal[:, lead_idx]
        fs = float(record.fs)

        draw_ecg(sig, img_path, fs)

        prompt = DRIFT_PROMPT if sample_type == "drift" else NOISE_PROMPT

        call1 = call_vlm_with_retry(
            client=client,
            model_name=model_name,
            image_path=img_path,
            prompt=prompt,
            sample_type=sample_type,
            max_retry=max_retry,
        )

        call2 = call_vlm_with_retry(
            client=client,
            model_name=model_name,
            image_path=img_path,
            prompt=prompt,
            sample_type=sample_type,
            max_retry=max_retry,
        )

        if not call1["ok"] or not call2["ok"]:
            return {
                "success": False,
                "payload": {
                    "ecg_id": ecg_id,
                    "lead_name": lead_name,
                    "type": sample_type,
                    "error_stage": "vlm_retry_failed",
                    "vlm_call_1": call1,
                    "vlm_call_2": call2
                }
            }

        res1 = call1["result"]
        res2 = call2["result"]

        if not validate_result(res1, sample_type) or not validate_result(res2, sample_type):
            return {
                "success": False,
                "payload": {
                    "ecg_id": ecg_id,
                    "lead_name": lead_name,
                    "type": sample_type,
                    "error_stage": "final_validation_failed",
                    "vlm_result_1": res1,
                    "vlm_result_2": res2
                }
            }

        return {
            "success": True,
            "payload": {
                "ecg_id": ecg_id,
                "lead_name": lead_name,
                "type": sample_type,
                "vlm_result_1": res1,
                "vlm_result_2": res2
            }
        }

    except Exception as e:
        return {
            "success": False,
            "payload": {
                "ecg_id": ecg_id,
                "lead_name": lead_name,
                "type": sample_type,
                "error_stage": "render_or_record_read",
                "error": str(e)
            }
        }

    finally:
        if img_path.exists():
            try:
                img_path.unlink()
            except Exception:
                pass


# =========================================================
# Main
# =========================================================
def main() -> None:
    args = parse_args()

    api_key = os.getenv("DASHSCOPE_API_KEY", "").strip()
    if not api_key:
        raise ValueError(
            "DASHSCOPE_API_KEY is not set.\n"
            "Please run:\n"
            'export DASHSCOPE_API_KEY="your_api_key"'
        )

    root_dir = Path(args.root_dir).resolve()
    meta_path = resolve_path(root_dir, args.meta_path)
    input_jsonl = resolve_path(root_dir, args.input_jsonl)
    records_dir = resolve_path(root_dir, args.records_dir)
    output_jsonl = resolve_path(root_dir, args.output_jsonl)
    error_jsonl = resolve_path(root_dir, args.error_jsonl)

    ensure_parent_dir(output_jsonl)
    ensure_parent_dir(error_jsonl)

    if not meta_path.exists():
        raise FileNotFoundError(f"Metadata CSV not found: {meta_path}")
    if not input_jsonl.exists():
        raise FileNotFoundError(f"Input JSONL not found: {input_jsonl}")
    if not records_dir.exists():
        raise FileNotFoundError(f"records500 directory not found: {records_dir}")

    client = OpenAI(
        api_key=api_key,
        base_url=args.base_url,
    )

    id_to_path = load_id_to_path_from_csv(meta_path)
    print(f"✅ Metadata loaded: {len(id_to_path)} ECG path mappings")

    data = load_jsonl(input_jsonl)
    print(f"📄 Total candidate samples: {len(data)}")

    done_set = load_done_set(output_jsonl)
    print(f"✅ Already completed (success only): {len(done_set)}")

    pending_data = []
    for item in data:
        ecg_id = str(item["ecg_id"]).zfill(5)
        lead_name = item["lead_name"]
        sample_type = item["type"]
        key = (ecg_id, lead_name, sample_type)

        if key not in done_set:
            pending_data.append(item)

    print(f"⏳ Pending samples: {len(pending_data)}")

    success_count = 0
    error_count = 0

    with open(output_jsonl, "a", encoding="utf-8") as fout, \
         open(error_jsonl, "a", encoding="utf-8") as ferr:

        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            futures = [
                executor.submit(
                    process_one,
                    item,
                    id_to_path,
                    records_dir,
                    client,
                    args.model_name,
                    args.max_retry,
                )
                for item in pending_data
            ]

            pbar = tqdm(total=len(futures))

            for future in as_completed(futures):
                try:
                    res = future.result()
                    if not res:
                        pbar.update(1)
                        continue

                    with write_lock:
                        if res["success"]:
                            fout.write(json.dumps(res["payload"], ensure_ascii=False) + "\n")
                            fout.flush()
                            success_count += 1
                        else:
                            ferr.write(json.dumps(res["payload"], ensure_ascii=False) + "\n")
                            ferr.flush()
                            error_count += 1

                except Exception as e:
                    err_obj = {
                        "error_stage": "future_result_exception",
                        "error": str(e)
                    }
                    with write_lock:
                        ferr.write(json.dumps(err_obj, ensure_ascii=False) + "\n")
                        ferr.flush()
                        error_count += 1

                pbar.update(1)

            pbar.close() 

    print("\n================ FINAL SUMMARY ================")
    print(f"✅ Success written: {success_count}")
    print(f"❌ Error written  : {error_count}")
    print(f"✅ Success file   : {output_jsonl}")
    print(f"❌ Error file     : {error_jsonl}")
    print("✅ Step 5 finished.")


if __name__ == "__main__":
    main()