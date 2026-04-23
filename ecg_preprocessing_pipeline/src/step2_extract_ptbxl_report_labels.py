#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Step 2: Extract structured ECG report labels from PTB-XL human-involved reports using LLMs.

This script reads the JSONL file generated in Step 1, applies a three-stage LLM
pipeline (LLM1 / LLM2 / LLM3), and writes structured label results.

Pipeline:
1. LLM1 independently parses the report
2. LLM2 independently parses the report
3. LLM3 arbitrates between LLM1 and LLM2

Input:
    {root_dir}/data/interim/ptbxl/ptbxl_human_report_empty_schema.jsonl
    {root_dir}/data/raw/ptbxl/SNOMED_labels.json

Output:
    output_path = root_dir / "outputs" / "ptbxl" / "step2_ptbxl_human_report_filled.jsonl"
    error_log_path = root_dir / "outputs" / "ptbxl" / "step2_ptbxl_human_report_error_log.jsonl"

Usage:
    export DEEPSEEK_API_KEY="your_api_key"
    python src/step2_extract_ptbxl_report_labels.py --root_dir /path/to/ecg_preprocessing_pipeline

Notes:
- Resume is supported: already processed ecg_id values in the output file will be skipped.
- Samples that fail processing in intermediate rounds are NOT immediately written to error_log.
- After all rounds finish, input/output are reconciled, and only samples still missing
  from output are written into the final error log.
- This version intentionally does NOT apply unmapped rule repair. Unmapped terms
  are preserved for later dedicated dual-LLM voting / repair steps.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests
from tqdm import tqdm


# =========================================================
# Global runtime config
# =========================================================
DEFAULT_MAX_RETRIES = 3
REQUEST_TIMEOUT = 90
RETRY_SLEEP = 2
API_RATE_SLEEP = 0.2
DEFAULT_MAX_WORKERS = 3
DEFAULT_RETRY_ROUNDS = 3

write_lock = threading.Lock()


# =========================================================
# Argument parsing
# =========================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract structured PTB-XL report labels using LLM1/LLM2/LLM3."
    )
    parser.add_argument(
        "--root_dir",
        type=str,
        required=True,
        help="Root directory of the ecg_preprocessing_pipeline project."
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=DEFAULT_MAX_WORKERS,
        help=f"Maximum number of worker threads (default: {DEFAULT_MAX_WORKERS})."
    )
    parser.add_argument(
        "--max_retries",
        type=int,
        default=DEFAULT_MAX_RETRIES,
        help=f"Maximum number of API retries per LLM call (default: {DEFAULT_MAX_RETRIES})."
    )
    parser.add_argument(
        "--retry_rounds",
        type=int,
        default=DEFAULT_RETRY_ROUNDS,
        help=f"Number of extra full retry rounds after the first pass (default: {DEFAULT_RETRY_ROUNDS})."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="deepseek-chat",
        help='LLM model name (default: "deepseek-chat").'
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Optional limit for test runs. Example: --max_samples 20"
    )
    return parser.parse_args()


# =========================================================
# Utility
# =========================================================
def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
    return data


def extract_json(text: str) -> Dict[str, Any]:
    """
    Try strict JSON parsing first; if it fails, extract the first JSON object block.
    """
    try:
        return json.loads(text)
    except Exception:
        pass

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except Exception:
            return {}

    return {}


def deduplicate_preserve_order(items: List[str]) -> List[str]:
    seen = set()
    result = []
    for x in items:
        if x not in seen:
            seen.add(x)
            result.append(x)
    return result


def clean_raw_list(raw_list: List[Any]) -> List[str]:
    cleaned = []
    for item in raw_list:
        text = str(item).strip().replace(" ", "")
        if text:
            cleaned.append(text)
    return deduplicate_preserve_order(cleaned)


def clean_string_list(items: List[Any]) -> List[str]:
    cleaned = []
    for item in items:
        text = str(item).strip()
        if text:
            cleaned.append(text)
    return deduplicate_preserve_order(cleaned)


def load_done_ids(output_path: Path) -> set:
    done_ids = set()
    if output_path.exists():
        with open(output_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    ecg_id = obj.get("ecg_id")
                    if ecg_id is not None:
                        done_ids.add(ecg_id)
                except Exception:
                    continue
    return done_ids


def load_output_id_counts(output_path: Path) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    if output_path.exists():
        with open(output_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    ecg_id = obj.get("ecg_id")
                    if ecg_id is not None:
                        counts[ecg_id] = counts.get(ecg_id, 0) + 1
                except Exception:
                    continue
    return counts


# =========================================================
# Validation
# =========================================================
def validate_llm12(obj: Dict[str, Any], label_set: set) -> bool:
    try:
        assert isinstance(obj, dict)
        assert "report_cn" in obj
        assert "report_label" in obj
        assert "explicit" in obj
        assert "uncertain" in obj
        assert "historical" in obj
        assert "old_MI" in obj

        rl = obj["report_label"]
        assert isinstance(rl, dict)
        assert isinstance(rl["raw"], list)
        assert isinstance(rl["mapped"], list)
        assert isinstance(rl["unmapped"], list)

        for x in rl["mapped"]:
            assert x in label_set

        assert isinstance(obj["explicit"], bool)
        assert isinstance(obj["uncertain"], bool)
        assert isinstance(obj["historical"], bool)
        assert isinstance(obj["old_MI"], bool)

        return True
    except Exception:
        return False


def validate_llm3(obj: Dict[str, Any], label_set: set) -> bool:
    try:
        assert isinstance(obj, dict)
        assert "final" in obj
        assert "status" in obj
        assert "reason" in obj

        final = obj["final"]
        assert validate_llm12(final, label_set)

        assert isinstance(obj["status"], str)
        assert isinstance(obj["reason"], str)

        return True
    except Exception:
        return False


# =========================================================
# API
# =========================================================
def call_api(
    prompt: str,
    api_key: str,
    model: str,
    max_retries: int,
) -> str:
    url = "https://api.deepseek.com/v1/chat/completions"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0
    }

    for _ in range(max_retries):
        try:
            response = requests.post(
                url,
                headers=headers,
                json=payload,
                timeout=REQUEST_TIMEOUT
            )

            try:
                res = response.json()
            except Exception:
                time.sleep(RETRY_SLEEP)
                continue

            if "choices" in res:
                time.sleep(API_RATE_SLEEP)
                return res["choices"][0]["message"]["content"].strip()

            time.sleep(RETRY_SLEEP)

        except Exception:
            time.sleep(RETRY_SLEEP)

    return ""


# =========================================================
# Prompt builders
# =========================================================
def build_llm12_prompt(report: str, label_list: List[str]) -> str:
    return f"""
请严格完成以下任务，并且只输出一个合法 JSON，不要输出任何解释、前后缀、markdown。

⚠️ 输出必须是标准 JSON：
- 不允许出现 ```json ``` 或任何额外文字
- 不允许输出多个 JSON
- 不允许字段缺失
- 不允许字段重命名

任务：
你要读取 ECG report，并完成以下四件事：

1. 翻译成标准化医学中文语义，填写到 "report_cn"
   - 必须严格基于原文
   - 不允许添加原文没有的信息
   - 不允许丢失原文信息
   - 不允许自行推理

2. 标签提取到 report_label.raw
   - raw 是从 report_cn 中切出来的医学标签或医学特征
   - 可以包括诊断、节律、波形/电图特征、传导异常、缺血/梗死相关描述等
   - 允许表达如：异常、改变、低电压等
   - 例如："周围低电压"、"窦性心律"、"正常心电图"、"QRS异常"、"ST段压低"
   - 只能使用 report_cn 中已有的信息，不允许自己加工推理或改写原意

3. 把 raw 里的内容映射到 report_label.mapped
   - mapped 只能从下面标签库中选择
   - 不允许自己创造新标签
   - 不允许修改标签名称
   - 必须选择最贴切的标准标签
   - 如果 raw 中某个特征在标签库里找不到合适标准标签，则不要强行匹配
   - 2:1传导,3:1传导不用映射

4. 把无法匹配到标签库的 raw 内容放入 report_label.unmapped

5. 判断以下四项，并填写布尔值 true/false：
   - explicit: 是否有明确诊断/明确特征
   - uncertain: 是否包含“可能、怀疑、不除外、nicht auszuschliessen、möglich、wahrscheinlich、verdacht”等不确定表达
   - historical: 是否记录既往信息/与之前对比/旧病史/年龄不明等时间性历史信息
   - old_MI: 是否明确提到陈旧性心肌梗死、旧性心梗、infarkt alt、wahrscheinlich alt、alter unbest. 等

=====================
【重点规则（必须严格遵守）】
=====================

A. 当前 vs 历史
- 出现：既往、之前、previous、compared with 等 → 属于历史信息
- 历史诊断绝对不能输出
- 只保留本次检查的结论

---------------------

B. 否定信息
- 出现：未见、无、未发现、no、without、absent
- 表示“没有该异常”
- 对应诊断绝对不能输出

示例：
"未见T波倒置" → ❌ 不能输出 "T波倒置"

---------------------

C. 不确定诊断（必须删除）
- 出现：可能、考虑、不排除、疑似、possible、likely、verdacht
- 不确定诊断不能进入 raw / mapped / unmapped

---------------------

D. 禁止医学推理
- 只能提取 report 明确写出的内容
- 不允许推理新诊断

示例：
T波异常 → ❌ 不能推理为心肌梗死

---------------------

E. 矛盾诊断
- 如：完全性 vs 不完全性传导阻滞
- 只能保留一个（按 report 明确描述）

---------------------

F. 陈旧性心肌梗死（重要）
- 所有“陈旧性/旧性/慢性心肌梗死”
👉 必须统一映射成列表里的 “陈旧性心肌梗死”
👉 不允许下壁陈旧性心肌梗死，映射成“下壁心肌梗死","陈旧性心肌梗死"。
只有明确的心肌梗死可以映射到心肌梗死相关的诊断，
所有的陈旧性心肌梗死，无论是下壁，侧壁，高侧壁等等，统一映射成陈旧性心肌梗死

---------------------

G. 正常 vs 异常
- "正常心电图" 只能和 "窦性心律" 同时出现
- 如果存在异常 → 删除 "正常心电图"

---------------------

H. 优先级
- 明确诊断 > 否定信息 > 推测信息
- 只保留明确、肯定的当前诊断

- 以下内容不进入 mapped / unmapped：
  - 心电轴正常
  - 无异常
  - 正常变异

---------------------

I. 当前已有诊断不可信
- 必须基于 report 重新判断
- 不允许照抄已有标签

---------------------

J. 如果 report 存在其余正常心电图的相关描述
- "正常心电图" 的标签禁止放进 mapped 里

⚠️ 重要约束：
- mapped 中的每个元素必须来自标签库
- unmapped 不能包含 mapped 中已有内容
- raw / mapped / unmapped 不能为空时才填写，否则返回空数组 []

标签库如下（mapped 只能从这里选）：
{label_list}

输出 JSON 格式必须严格为：
{{
  "report_cn": "",
  "report_label": {{
    "raw": [],
    "mapped": [],
    "unmapped": []
  }},
  "explicit": false,
  "uncertain": false,
  "historical": false,
  "old_MI": false
}}

ECG report:
{report}
"""


def build_llm3_prompt(
    report: str,
    llm1_obj: Dict[str, Any],
    llm2_obj: Dict[str, Any],
    label_list: List[str],
) -> str:
    llm1_text = json.dumps(llm1_obj, ensure_ascii=False)
    llm2_text = json.dumps(llm2_obj, ensure_ascii=False)

    return f"""
你是 ECG 报告仲裁器。

⚠️ 以下规则优先级最高，必须严格执行
⚠️ 只输出一个合法 JSON，不允许任何解释、前后缀或 markdown

=====================
【输入】
=====================

原始 report:
{report}

LLM1:
{llm1_text}

LLM2:
{llm2_text}

=====================
【任务】
=====================

你需要基于 report，对 LLM1 和 LLM2 的结果进行裁决，输出最终标准结果。

=====================
【裁决规则】
=====================

1. 如果 LLM1 和 LLM2 的 raw、mapped、unmapped 完全一致
→ status = "pass"

2. 如果 LLM1 和 LLM2 高度一致（仅表达不同、语义一致）
→ status = "high_consistency"

3. 如果存在明显差异
→ 必须重新基于 report 独立判断
→ 可以不同于 LLM1 / LLM2
→ status = "conflict"
→ 必须给出简短 reason

=====================
【医学规则（必须执行）】
=====================

A. 当前 vs 历史
- 既往/previous 等历史信息不能输出

B. 否定信息
- 未见/无/no 等不能生成诊断

C. 不确定诊断
- 可能/考虑/不排除 等不能进入任何字段

D. 禁止推理
- 只能提取 report 明确写出的内容

E. 矛盾诊断
- 只能保留一个

F. 陈旧性心肌梗死
- 所有旧性/陈旧性心梗统一映射为："陈旧性心肌梗死"

G. 正常 vs 异常
- 如果存在异常 → 删除“正常心电图”
- "正常心电图" 只能与 "窦性心律" 共存

H. 优先级
- 明确诊断 > 否定信息 > 推测信息

- 以下内容不进入 mapped / unmapped：
  心电轴正常、无异常、正常变异

I. 不允许照抄 LLM1/LLM2
- 必须基于 report 重新判断

=====================
【输出要求】
=====================

- final.report_cn 必须严格来自 report
- raw：医学表达（来自 report）
- mapped：只能从标签库中选择
- unmapped：无法匹配标签库的表达

- mapped 中的元素必须全部来自标签库
- unmapped 不能包含 mapped 中已有内容

- 所有数组必须为 list
- 如果为空必须返回 []

- 所有布尔字段必须为 true/false

标签库如下（mapped 只能从这里选）：
{label_list}

=====================
【输出格式】
=====================

{{
  "final": {{
    "report_cn": "",
    "report_label": {{
      "raw": [],
      "mapped": [],
      "unmapped": []
    }},
    "explicit": false,
    "uncertain": false,
    "historical": false,
    "old_MI": false
  }},
  "status": "pass",
  "reason": ""
}}
"""


# =========================================================
# LLM runners
# =========================================================
def run_llm12(
    report: str,
    label_list: List[str],
    label_set: set,
    api_key: str,
    model: str,
    max_retries: int,
) -> Optional[Dict[str, Any]]:
    prompt = build_llm12_prompt(report, label_list)

    for _ in range(max_retries):
        text = call_api(
            prompt=prompt,
            api_key=api_key,
            model=model,
            max_retries=max_retries,
        )
        if not text:
            time.sleep(RETRY_SLEEP)
            continue

        obj = extract_json(text)
        if validate_llm12(obj, label_set):
            rl = obj.get("report_label", {})
            rl["raw"] = clean_raw_list(rl.get("raw", []))
            rl["mapped"] = clean_string_list(rl.get("mapped", []))
            rl["unmapped"] = clean_string_list(rl.get("unmapped", []))
            obj["report_label"] = rl
            return obj

        time.sleep(RETRY_SLEEP)

    return None


def run_llm3(
    report: str,
    llm1_obj: Dict[str, Any],
    llm2_obj: Dict[str, Any],
    label_list: List[str],
    label_set: set,
    api_key: str,
    model: str,
    max_retries: int,
) -> Dict[str, Any]:
    if llm1_obj == llm2_obj:
        return {"final": llm1_obj, "status": "pass", "reason": ""}

    if (
        llm1_obj["report_label"]["mapped"] == llm2_obj["report_label"]["mapped"]
        and llm1_obj["report_label"]["unmapped"] == llm2_obj["report_label"]["unmapped"]
    ):
        return {
            "final": llm1_obj,
            "status": "high_consistency",
            "reason": "核心标签一致"
        }

    prompt = build_llm3_prompt(report, llm1_obj, llm2_obj, label_list)

    for _ in range(max_retries):
        text = call_api(
            prompt=prompt,
            api_key=api_key,
            model=model,
            max_retries=max_retries,
        )
        if not text:
            time.sleep(RETRY_SLEEP)
            continue

        obj = extract_json(text)
        if validate_llm3(obj, label_set):
            final_rl = obj["final"].get("report_label", {})
            final_rl["raw"] = clean_raw_list(final_rl.get("raw", []))
            final_rl["mapped"] = clean_string_list(final_rl.get("mapped", []))
            final_rl["unmapped"] = clean_string_list(final_rl.get("unmapped", []))
            obj["final"]["report_label"] = final_rl
            return obj

        time.sleep(RETRY_SLEEP)

    pick = llm1_obj if len(llm1_obj["report_label"]["mapped"]) >= len(llm2_obj["report_label"]["mapped"]) else llm2_obj

    return {
        "final": pick,
        "status": "conflict",
        "reason": "fallback"
    }


# =========================================================
# Processing
# =========================================================
def process_one(
    data: Dict[str, Any],
    label_list: List[str],
    label_set: set,
    api_key: str,
    model: str,
    max_retries: int,
) -> Tuple[str, Dict[str, Any]]:
    ecg_id = data["ecg_id"]
    report = data.get("report", "")

    try:
        item = copy.deepcopy(data)

        llm1 = run_llm12(
            report=report,
            label_list=label_list,
            label_set=label_set,
            api_key=api_key,
            model=model,
            max_retries=max_retries,
        )
        llm2 = run_llm12(
            report=report,
            label_list=label_list,
            label_set=label_set,
            api_key=api_key,
            model=model,
            max_retries=max_retries,
        )

        if llm1 is None or llm2 is None:
            raise ValueError("LLM1/LLM2 failed")

        llm3 = run_llm3(
            report=report,
            llm1_obj=llm1,
            llm2_obj=llm2,
            label_list=label_list,
            label_set=label_set,
            api_key=api_key,
            model=model,
            max_retries=max_retries,
        )

        item["llm1"] = llm1
        item["llm2"] = llm2
        item["llm3"] = llm3

        return ("ok", item)

    except Exception as e:
        return (
            "error",
            {
                "ecg_id": ecg_id,
                "report": report,
                "error": str(e)
            }
        )


# =========================================================
# Final reconciliation
# =========================================================
def build_final_error_log(
    data_list: List[Dict[str, Any]],
    output_path: Path,
    error_log_path: Path,
    last_error_map: Dict[str, Dict[str, Any]],
) -> Tuple[int, int]:
    """
    Reconcile input vs output after all rounds.
    Only samples missing from output are written to the final error log.
    """
    final_done_ids = load_done_ids(output_path)
    missing_items = [d for d in data_list if d["ecg_id"] not in final_done_ids]

    with open(error_log_path, "w", encoding="utf-8") as ferr:
        for d in missing_items:
            ecg_id = d["ecg_id"]
            report = d.get("report", "")
            err_obj = last_error_map.get(
                ecg_id,
                {
                    "ecg_id": ecg_id,
                    "report": report,
                    "error": "missing_from_output_after_all_rounds"
                }
            )
            ferr.write(json.dumps(err_obj, ensure_ascii=False) + "\n")

    return len(final_done_ids), len(missing_items)


# =========================================================
# Main
# =========================================================
def main() -> None:
    args = parse_args()
    root_dir = Path(args.root_dir).resolve()

    api_key = os.getenv("DEEPSEEK_API_KEY", "").strip()
    if not api_key:
        raise ValueError(
            "DEEPSEEK_API_KEY is not set.\n"
            "Please run:\n"
            'export DEEPSEEK_API_KEY="your_api_key"'
        )

    input_path = root_dir / "data" / "interim" / "ptbxl" / "ptbxl_human_report_empty_schema.jsonl"
    label_path = root_dir / "data" / "raw" / "ptbxl" / "SNOMED_labels.json"

    output_path = root_dir / "outputs" / "ptbxl" / "step2_ptbxl_human_report_filled.jsonl"
    error_log_path = root_dir / "outputs" / "ptbxl" / "step2_ptbxl_human_report_error_log.jsonl"

    ensure_parent_dir(output_path)
    ensure_parent_dir(error_log_path)

    if not input_path.exists():
        raise FileNotFoundError(f"Input JSONL not found: {input_path}")
    if not label_path.exists():
        raise FileNotFoundError(f"Label JSON not found: {label_path}")

    label_list = load_json(label_path)
    if not isinstance(label_list, list):
        raise ValueError(f"Label JSON must be a list, but got: {type(label_list)}")
    label_set = set(label_list)

    data_list = load_jsonl(input_path)
    if args.max_samples is not None:
        data_list = data_list[:args.max_samples]

    print(f"📄 Total input samples: {len(data_list)}")

    # 保存“最后一次失败信息”，最终只给仍未成功的样本写 error_log
    last_error_map: Dict[str, Dict[str, Any]] = {}

    total_rounds = 1 + args.retry_rounds

    for round_idx in range(total_rounds):
        done_ids = load_done_ids(output_path)
        remaining_data = [d for d in data_list if d["ecg_id"] not in done_ids]
        remaining_count = len(remaining_data)

        print(f"\n================ ROUND {round_idx + 1}/{total_rounds} ================")
        print(f"🔁 Already completed: {len(done_ids)}")
        print(f"⏳ Remaining to process: {remaining_count}")

        if remaining_count == 0:
            print("✅ All samples have been completed. Stop early.")
            break

        round_ok = 0
        round_error = 0

        with open(output_path, "a", encoding="utf-8") as fout:
            with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
                futures = [
                    executor.submit(
                        process_one,
                        d,
                        label_list,
                        label_set,
                        api_key,
                        args.model,
                        args.max_retries,
                    )
                    for d in remaining_data
                ]

                for future in tqdm(as_completed(futures), total=len(futures)):
                    status, obj = future.result()

                    with write_lock:
                        if status == "ok":
                            ecg_id = obj["ecg_id"]

                            # 双保险：写入前再次核对，避免极少数重复写入
                            current_done_ids = load_done_ids(output_path)
                            if ecg_id not in current_done_ids:
                                fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
                                fout.flush()
                                round_ok += 1

                            # 一旦成功，删除历史错误缓存
                            if ecg_id in last_error_map:
                                del last_error_map[ecg_id]

                        else:
                            ecg_id = obj["ecg_id"]
                            last_error_map[ecg_id] = obj
                            round_error += 1

        print(f"✅ Round {round_idx + 1} finished: ok={round_ok}, transient_error={round_error}")

    # 最终对账：只保留 output 中仍然缺失的样本
    final_done, final_missing = build_final_error_log(
        data_list=data_list,
        output_path=output_path,
        error_log_path=error_log_path,
        last_error_map=last_error_map,
    )

    # 额外检查 output 是否有重复 ecg_id
    output_counts = load_output_id_counts(output_path)
    duplicate_count = sum(1 for _, c in output_counts.items() if c > 1)

    print("\n================ FINAL SUMMARY ================")
    print(f"✅ Total completed: {final_done}")
    print(f"⏳ Still missing from output: {final_missing}")
    print(f"🧾 Final error log written: {error_log_path}")
    print(f"🔍 Duplicate ecg_id in output: {duplicate_count}")
    print("✅ Step 2 finished.")


if __name__ == "__main__":
    main()