#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Step 3.3: Flatten consensus unmapped terms and fill them with LLM mapping results.

Input:
    {root_dir}/outputs/ptbxl/step3_2_ptbxl_consensus_unmapped.jsonl
    {root_dir}/data/raw/ptbxl/SNOMED_labels.json

Output:
    {root_dir}/outputs/ptbxl/step3_3_ptbxl_consensus_unmapped_terms_filled.jsonl
    {root_dir}/outputs/ptbxl/step3_3_ptbxl_consensus_unmapped_terms_stats.json

Pipeline:
1. Read step3_2 consensus unmapped file
2. Flatten all unmapped terms into unique term list
3. Create / resume a term-level JSONL:
   {"input": "...", "llm1": {"mapped": []}, "llm2": {"mapped": []}, "llm3": {"mapped": [], "status": "pass", "reason": ""}}
4. Fill with LLM1 / LLM2 / LLM3
5. Save term-level mapped results

Notes:
- Resume supported by `input`
- Only labels in label_list are allowed
- Temperature fixed at 0
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Set

from openai import OpenAI
from tqdm import tqdm


# =========================================================
# Global config
# =========================================================
DEFAULT_MAX_WORKERS = 6
MAX_API_RETRIES = 3
SLEEP_BETWEEN_RETRIES = 1.0
RANDOM_RATE_SLEEP_MIN = 0.3
RANDOM_RATE_SLEEP_MAX = 0.8

write_lock = threading.Lock()


# =========================================================
# Argument parsing
# =========================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Flatten consensus unmapped terms and fill them using LLM mapping."
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
        help="Optional custom input path. Default: outputs/ptbxl/step3_2_ptbxl_consensus_unmapped.jsonl"
    )
    parser.add_argument(
        "--label_path",
        type=str,
        default=None,
        help="Optional custom label path. Default: data/raw/ptbxl/SNOMED_labels.json"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Optional custom output path. Default: outputs/ptbxl/step3_3_ptbxl_consensus_unmapped_terms_filled.jsonl"
    )
    parser.add_argument(
        "--stats_path",
        type=str,
        default=None,
        help="Optional custom stats path. Default: outputs/ptbxl/step3_3_ptbxl_consensus_unmapped_terms_stats.json"
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=DEFAULT_MAX_WORKERS,
        help=f"Maximum worker threads (default: {DEFAULT_MAX_WORKERS})"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="deepseek-chat",
        help='Model name for all three calls (default: "deepseek-chat")'
    )
    parser.add_argument(
        "--max_terms",
        type=int,
        default=None,
        help="Optional limit for debug runs, e.g. --max_terms 100"
    )
    return parser.parse_args()


# =========================================================
# IO helpers
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


def write_json(path: Path, obj: Dict[str, Any]) -> None:
    ensure_parent_dir(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_done_inputs(output_path: Path) -> Set[str]:
    done = set()
    if not output_path.exists():
        return done

    with open(output_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
                term = str(obj.get("input", "")).strip()
                if term:
                    done.add(term)
            except Exception:
                continue
    return done


# =========================================================
# Utility
# =========================================================
def dedup_keep_order(items: List[str]) -> List[str]:
    seen = set()
    out = []
    for x in items:
        x = str(x).strip()
        if not x:
            continue
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def safe_json_load(text: str) -> Dict[str, Any] | None:
    text = text.strip()
    text = re.sub(r"```json|```", "", text)

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except Exception:
            return None
    return None


def normalize_mapped(mapped: List[Any], label_set: Set[str]) -> List[str]:
    cleaned = []
    seen = set()
    for x in mapped:
        s = str(x).strip()
        if not s:
            continue
        if s in label_set and s not in seen:
            seen.add(s)
            cleaned.append(s)
    return cleaned


def flatten_unique_terms(data_list: List[Dict[str, Any]]) -> List[str]:
    all_terms = []
    for item in data_list:
        unmapped = item.get("unmapped", [])
        if isinstance(unmapped, list):
            for term in unmapped:
                term = str(term).strip()
                if term:
                    all_terms.append(term)
    return dedup_keep_order(all_terms)


# =========================================================
# Prompts
# =========================================================
SYSTEM_PROMPT = """你是心电图（ECG）医学标准化专家。

你的任务是：将输入的心电图文本，映射为“标签列表中的一个或多个标准标签”。

⚠️ 你是分类器，不是解释器。
⚠️ 你只能做“表达归一”，不能做“医学推理”。

====================
【输出规则（必须严格遵守）】

1. 只能输出 JSON
2. 不允许输出任何解释、说明、分析或多余文字
3. mapped 必须是列表（list）
4. 输出结构必须为：

{
  "mapped": []
}

====================
【标签规则】

1. 只能使用提供的标签列表中的标签
2. 不允许创造新标签
3. mapped 中的每个元素必须：
   - 完全等于标签列表中的某一项
   - 不能是原始文本片段
4. 如果无法明确匹配 → mapped = []

====================
【核心原则（非常重要）】

❗禁止推理、补全、升级、猜测：

- 不允许根据经验推断更具体诊断
- 不允许从“2:1”推断“II型”
- 不允许从“可能”推断“确定诊断”
- 不允许进行医学解释性升级
- 不允许扩展含义

👉 你只能做：
“文本归一（normalization）”，而不是“诊断推理（diagnosis inference）”

示例：

输入：2:1房室传导阻滞
❌ 错误：二度II型房室传导阻滞
✅ 正确：房室传导阻滞

====================
【语义规则（必须执行）】

1️⃣ 去噪：
- 删除导联信息（V1、V2、III、aVF等）
- 删除部位信息（前壁、下壁、外侧壁等）

2️⃣ 同义归一：
- T波负向 = T波倒置
- 左心电轴 = 电轴左偏 = 左轴偏移

3️⃣ 细分归一：
- 前壁缺血 → 心肌缺血
- 高外侧缺血 → 心肌缺血

4️⃣ 拆分规则（必须执行）：
- QRS(T)异常 → 异常QRS + T波异常
- QRS(T)波异常 → 异常QRS + T波异常
- ST-T异常 → ST段异常 + T波异常

👉 拆分后必须保留所有有效标签

====================
【去过拟合规则（非常重要）】

当存在“更具体标签”和“更通用标签”时：

👉 只能选择：
“输入文本明确支持的、语义最贴近的标签”

规则：
- “异常 / 改变” → 只能映射为“异常类标签”
- 不允许升级为：
  抬高 / 倒置 / 低电压 等具体形态标签

示例：
输入：T波异常
❌ 错误：T波倒置
✅ 正确：T波异常

====================
【同类标签选择规则】

如果多个标签属于同一类别：

👉 只允许选择一个最贴近输入的标签

示例：
- T波倒置 vs T波异常 → 选择最贴近输入表达的
- 不允许同时输出两个同类标签

====================
【最终检查（必须执行）】

在输出前，必须检查：

1. mapped 是否为 list
2. 每个元素是否在标签列表中
3. 是否存在推理或补全行为

如果任何一项不满足 → 输出：

{
  "mapped": []
}
"""


SYSTEM_PROMPT_JUDGE = """你是心电图（ECG）标签仲裁专家。

你的任务是：在两个模型结果之间进行判断，而不是重新做分类。

⚠️ 你是“仲裁器”，不是“标签生成器”。

====================
【你的职责】

你必须：

1. 判断两个结果是否一致
2. 如果一致 → 选择其中一个
3. 如果冲突 → 独立重新判断

====================
【严格规则】

1. 不允许直接复制 A 或 B（在冲突时）
2. 不允许融合 A 和 B
3. 不允许进行医学推理或升级
4. 只能使用标签列表中的标签

====================
【mapped 规则】

- 必须是 list
- 每个元素必须来自标签列表
- 不合法 → 返回 []

====================
【标签规则】

1. 只能使用提供的标签列表中的标签
2. 不允许创造新标签
3. mapped 中的每个元素必须：
   - 完全等于标签列表中的某一项
   - 不能是原始文本片段
4. 如果无法明确匹配 → mapped = []

====================
【核心原则（非常重要）】

❗禁止推理、补全、升级、猜测：

- 不允许根据经验推断更具体诊断
- 不允许从“2:1”推断“II型”
- 不允许从“可能”推断“确定诊断”
- 不允许进行医学解释性升级
- 不允许扩展含义

👉 你只能做：
“文本归一（normalization）”，而不是“诊断推理（diagnosis inference）”

====================
【最终输出】

{
  "status": "高度一致 或 冲突",
  "mapped": [],
  "reason": ""
}

⚠️ 只能输出 JSON
"""


# =========================================================
# LLM callers
# =========================================================
def build_basic_user_prompt(item: str, label_text: str) -> str:
    return f"""
输入：
"{item}"

标签列表：
{label_text}

输出：
{{"mapped": []}}

⚠️ 只能输出 JSON
"""


def build_judge_user_prompt(item: str, m1: List[str], m2: List[str], label_text: str) -> str:
    return f"""
你是心电图（ECG）标签仲裁专家。

你的任务是：在两个模型结果之间进行判断，而不是生成自由答案。

====================
【输入】

原始文本：
"{item}"

标签列表：
{label_text}

模型A输出：
{m1}

模型B输出：
{m2}

====================
【输出格式（必须严格遵守）】

只允许输出 JSON：

{{
  "status": "高度一致 或 冲突",
  "mapped": [],
  "reason": ""
}}

====================
【仲裁规则（必须严格执行）】

1️⃣ 一致性判断：

如果满足以下任一情况 → status = "高度一致"

- 两者完全相同
- 或者语义等价

👉 此时：
- 选择“更标准 / 更归一 / 更通用”的那个
- mapped 只能选一个结果（A或B之一）
- 不允许自己生成新结果

====================

2️⃣ 冲突判断：

如果满足以下情况 → status = "冲突"

- 标签完全不同类别
- 或存在明显推理升级
- 或无法判断哪个更合理

👉 此时必须：
- 完全忽略 A 和 B
- 基于原始文本独立判断
- 不允许照抄 A 或 B

====================

【关键限制（非常重要）】

❗禁止行为：

- 不允许直接复制 A 或 B（在冲突时）
- 不允许融合 A 和 B
- 不允许补全、推理、升级标签
- 不允许输出标签列表之外的内容

====================

【mapped 规则】

1. mapped 必须是列表
2. 只能包含标签列表中的标签
3. 如果无法判断 → mapped = []

====================

【reason 规则】

- 必须用一句中文说明原因
- 简短，不解释医学

示例：
- "语义一致，选择更标准表达"
- "两者冲突，重新判断"

====================

⚠️ 最终要求：

你必须先判断一致性，再决定行为
你不能跳过这个步骤

⚠️ 只能输出 JSON，不能有任何额外内容
"""


def call_llm_basic(client: OpenAI, model: str, item: str, label_text: str, label_set: Set[str]) -> List[str]:
    user_prompt = build_basic_user_prompt(item, label_text)

    for _ in range(MAX_API_RETRIES):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0
            )

            text = response.choices[0].message.content
            res = safe_json_load(text)

            time.sleep(random.uniform(RANDOM_RATE_SLEEP_MIN, RANDOM_RATE_SLEEP_MAX))

            if res and isinstance(res.get("mapped"), list):
                return normalize_mapped(res["mapped"], label_set)

        except Exception:
            pass

        time.sleep(SLEEP_BETWEEN_RETRIES)

    return []


def call_llm_judge(
    client: OpenAI,
    model: str,
    item: str,
    m1: List[str],
    m2: List[str],
    label_text: str,
    label_set: Set[str]
) -> Dict[str, Any]:
    user_prompt = build_judge_user_prompt(item, m1, m2, label_text)

    for _ in range(MAX_API_RETRIES):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT_JUDGE},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0
            )

            text = response.choices[0].message.content
            res = safe_json_load(text)

            time.sleep(random.uniform(RANDOM_RATE_SLEEP_MIN, RANDOM_RATE_SLEEP_MAX))

            if res and isinstance(res, dict):
                mapped = normalize_mapped(res.get("mapped", []), label_set)
                return {
                    "status": str(res.get("status", "")).strip(),
                    "mapped": mapped,
                    "reason": str(res.get("reason", "")).strip()
                }

        except Exception:
            pass

        time.sleep(SLEEP_BETWEEN_RETRIES)

    return {"status": "冲突", "mapped": [], "reason": "失败"}


# =========================================================
# Core processing
# =========================================================
def process_one_term(
    client: OpenAI,
    model: str,
    term: str,
    label_text: str,
    label_set: Set[str],
) -> Dict[str, Any]:
    m1 = call_llm_basic(client, model, term, label_text, label_set)
    m2 = call_llm_basic(client, model, term, label_text, label_set)

    if set(m1) == set(m2):
        return {
            "input": term,
            "llm1": {"mapped": m1},
            "llm2": {"mapped": m2},
            "llm3": {
                "mapped": m1,
                "status": "pass",
                "reason": ""
            }
        }

    judge = call_llm_judge(client, model, term, m1, m2, label_text, label_set)
    m3 = normalize_mapped(judge.get("mapped", []), label_set)

    return {
        "input": term,
        "llm1": {"mapped": m1},
        "llm2": {"mapped": m2},
        "llm3": {
            "mapped": m3,
            "status": judge.get("status", ""),
            "reason": judge.get("reason", "")
        }
    }


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
            'Please run: export DEEPSEEK_API_KEY="your_api_key"'
        )

    input_path = (
        Path(args.input_path).resolve()
        if args.input_path
        else root_dir / "outputs" / "ptbxl" / "step3_2_ptbxl_consensus_unmapped.jsonl"
    )
    label_path = (
        Path(args.label_path).resolve()
        if args.label_path
        else root_dir / "data" / "raw" / "ptbxl" / "SNOMED_labels.json"
    )
    output_path = (
        Path(args.output_path).resolve()
        if args.output_path
        else root_dir / "outputs" / "ptbxl" / "step3_3_ptbxl_consensus_unmapped_terms_filled.jsonl"
    )
    stats_path = (
        Path(args.stats_path).resolve()
        if args.stats_path
        else root_dir / "outputs" / "ptbxl" / "step3_3_ptbxl_consensus_unmapped_terms_stats.json"
    )

    ensure_parent_dir(output_path)
    ensure_parent_dir(stats_path)

    if not input_path.exists():
        raise FileNotFoundError(f"Input not found: {input_path}")
    if not label_path.exists():
        raise FileNotFoundError(f"Label file not found: {label_path}")

    step3_2_data = load_jsonl(input_path)
    label_list = load_json(label_path)

    if not isinstance(label_list, list):
        raise ValueError(f"Label JSON must be a list, got: {type(label_list)}")

    label_list = dedup_keep_order([str(x).strip() for x in label_list if str(x).strip()])
    label_set = set(label_list)
    label_text = "，".join(label_list)

    all_terms = flatten_unique_terms(step3_2_data)

    if args.max_terms is not None:
        all_terms = all_terms[:args.max_terms]

    done_inputs = load_done_inputs(output_path)
    pending_terms = [x for x in all_terms if x not in done_inputs]

    print(f"📄 Total unique flattened unmapped terms : {len(all_terms)}")
    print(f"✅ Already done                          : {len(done_inputs)}")
    print(f"⏳ Pending                               : {len(pending_terms)}")

    client = OpenAI(
        api_key=api_key,
        base_url="https://api.deepseek.com"
    )

    status_counts: Dict[str, int] = {}
    total_written = 0

    with open(output_path, "a", encoding="utf-8") as f_out:
        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            futures = {
                executor.submit(
                    process_one_term,
                    client,
                    args.model,
                    term,
                    label_text,
                    label_set,
                ): term
                for term in pending_terms
            }

            for future in tqdm(as_completed(futures), total=len(futures)):
                term = futures[future]

                try:
                    result = future.result()
                except Exception:
                    result = {
                        "input": term,
                        "llm1": {"mapped": []},
                        "llm2": {"mapped": []},
                        "llm3": {"mapped": [], "status": "异常", "reason": ""}
                    }

                with write_lock:
                    f_out.write(json.dumps(result, ensure_ascii=False) + "\n")
                    f_out.flush()

                total_written += 1
                st = str(result.get("llm3", {}).get("status", "")).strip()
                status_counts[st] = status_counts.get(st, 0) + 1

                time.sleep(random.uniform(0.2, 0.5))

    stats = {
        "total_unique_flattened_terms": len(all_terms),
        "already_done_before_this_run": len(done_inputs),
        "pending_terms_this_run": len(pending_terms),
        "written_this_run": total_written,
        "llm3_status_counts_this_run": status_counts,
        "input_path": str(input_path),
        "label_path": str(label_path),
        "output_path": str(output_path),
    }

    write_json(stats_path, stats)

    print("\n================ STEP 3.3 SUMMARY ================")
    print(f"📄 Total unique flattened terms : {len(all_terms)}")
    print(f"✅ Already done                 : {len(done_inputs)}")
    print(f"🆕 Written this run             : {total_written}")
    print(f"📁 Output JSONL                 : {output_path}")
    print(f"📊 Stats JSON                   : {stats_path}")
    print("\n[LLM3 status counts this run]")
    for k, v in sorted(status_counts.items(), key=lambda x: x[0]):
        print(f"  - {k}: {v}")

    print("\n✅ Step 3.3 finished.")


if __name__ == "__main__":
    main()