#!/usr/bin/env bash
set -euo pipefail

# =========================================================
# PTB-XL Full Pipeline Runner
# =========================================================
# Usage:
#   bash run_ptbxl_pipeline.sh
#
# Fill these before running:
#   - ROOT_DIR
#   - DEEPSEEK_API_KEY
#   - DASHSCOPE_API_KEY
# =========================================================


# =========================================================
# 1. User config (fill these)
# =========================================================
ROOT_DIR="/Users/kongxiangyu/Documents/AI学习/生物医学深度学习模型/Heaformer/ecg_preprocessing_pipeline"

DEEPSEEK_API_KEY="sk-290a147be267404daf524e2472994049"
DASHSCOPE_API_KEY="sk-98c99501361c4d158cfd167121783fed"


# =========================================================
# 2. Runtime config
# =========================================================
PYTHON_BIN="python"

STEP2_MAX_WORKERS=3
STEP2_MAX_RETRIES=3
STEP2_RETRY_ROUNDS=3
STEP2_MODEL_NAME="deepseek-chat"

STEP3_3_MAX_WORKERS=6
STEP3_3_MODEL_NAME="deepseek-chat"

STEP4_MAX_WORKERS=6

STEP5_MAX_WORKERS=4
STEP5_MAX_RETRY=3
STEP5_MODEL_NAME="qwen3-vl-flash"
STEP5_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"


# =========================================================
# 3. Helper functions
# =========================================================
print_header() {
    echo
    echo "========================================================="
    echo "$1"
    echo "========================================================="
}

check_not_empty() {
    local name="$1"
    local value="$2"
    if [[ -z "$value" ]]; then
        echo "❌ $name is empty. Please fill it in the script first."
        exit 1
    fi
}

check_path_exists() {
    local path="$1"
    if [[ ! -e "$path" ]]; then
        echo "❌ Path not found: $path"
        exit 1
    fi
}

run_step() {
    local step_name="$1"
    shift

    print_header "🚀 Running ${step_name}"
    echo "Command:"
    printf '%q ' "$@"
    echo
    echo

    "$@"

    echo
    echo "✅ Finished ${step_name}"
}


# =========================================================
# 4. Pre-check
# =========================================================
print_header "🔍 Pre-check"

check_not_empty "ROOT_DIR" "$ROOT_DIR"
check_not_empty "DEEPSEEK_API_KEY" "$DEEPSEEK_API_KEY"
check_not_empty "DASHSCOPE_API_KEY" "$DASHSCOPE_API_KEY"

check_path_exists "$ROOT_DIR"
check_path_exists "$ROOT_DIR/src"
check_path_exists "$ROOT_DIR/data"

export DEEPSEEK_API_KEY="$DEEPSEEK_API_KEY"
export DASHSCOPE_API_KEY="$DASHSCOPE_API_KEY"

echo "✅ ROOT_DIR = $ROOT_DIR"
echo "✅ src/ exists"
echo "✅ data/ exists"
echo "✅ DEEPSEEK_API_KEY is set"
echo "✅ DASHSCOPE_API_KEY is set"


# =========================================================
# 5. Step 1
# =========================================================
run_step "Step 1 - Build PTB-XL human report schema" \
    "$PYTHON_BIN" "$ROOT_DIR/src/step1_build_ptbxl_human_report_schema.py" \
    --root_dir "$ROOT_DIR"


# =========================================================
# 6. Step 2
# =========================================================
run_step "Step 2 - Extract structured PTB-XL report labels" \
    "$PYTHON_BIN" "$ROOT_DIR/src/step2_extract_ptbxl_report_labels.py" \
    --root_dir "$ROOT_DIR" \
    --max_workers "$STEP2_MAX_WORKERS" \
    --max_retries "$STEP2_MAX_RETRIES" \
    --retry_rounds "$STEP2_RETRY_ROUNDS" \
    --model "$STEP2_MODEL_NAME"


# =========================================================
# 7. Step 3.1
# =========================================================
run_step "Step 3.1 - Filter PTB-XL report labels by LLM consensus" \
    "$PYTHON_BIN" "$ROOT_DIR/src/step3.1_filter_ptbxl_human_report_by_llm_consensus.py" \
    --root_dir "$ROOT_DIR"


# =========================================================
# 8. Step 3.2
# =========================================================
run_step "Step 3.2 - Extract consensus-supported unmapped terms" \
    "$PYTHON_BIN" "$ROOT_DIR/src/step3.2_extract_ptbxl_consensus_unmapped.py" \
    --root_dir "$ROOT_DIR"


# =========================================================
# 9. Step 3.3
# =========================================================
run_step "Step 3.3 - Flatten and fill consensus unmapped terms" \
    "$PYTHON_BIN" "$ROOT_DIR/src/step3.3_prepare_and_fill_consensus_unmapped_terms.py" \
    --root_dir "$ROOT_DIR" \
    --max_workers "$STEP3_3_MAX_WORKERS" \
    --model "$STEP3_3_MODEL_NAME"


# =========================================================
# 10. Step 3.4
# =========================================================
run_step "Step 3.4 - Build final PTB-XL report LLM labels" \
    "$PYTHON_BIN" "$ROOT_DIR/src/step3.4_build_ptxbl_report_label.py" \
    --root_dir "$ROOT_DIR"


# =========================================================
# 11. Step 4
# =========================================================
run_step "Step 4 - Compute ECG signal quality top5 candidates" \
    "$PYTHON_BIN" "$ROOT_DIR/src/step4_compute_signal_quality_top5.py" \
    --root_dir "$ROOT_DIR" \
    --meta_path data/raw/ptbxl/ptbxl_database.csv \
    --max_workers "$STEP4_MAX_WORKERS"


# =========================================================
# 12. Step 5
# =========================================================
run_step "Step 5 - VLM signal quality verification" \
    "$PYTHON_BIN" "$ROOT_DIR/src/step5_vlm_signal_quality_verification.py" \
    --root_dir "$ROOT_DIR" \
    --meta_path "data/raw/ptbxl/ptbxl_database.csv" \
    --input_jsonl "outputs/ptbxl/step4_signal_quality_top5.jsonl" \
    --records_dir "data/raw/ptbxl/records500" \
    --max_workers "$STEP5_MAX_WORKERS" \
    --max_retry "$STEP5_MAX_RETRY" \
    --model_name "$STEP5_MODEL_NAME" \
    --base_url "$STEP5_BASE_URL"


# =========================================================
# 13. Step 6
# =========================================================
run_step "Step 6 - Merge report labels with VLM quality flags" \
    "$PYTHON_BIN" "$ROOT_DIR/src/step6_merge_report_label_with_signal_quality.py" \
    --root_dir "$ROOT_DIR"


# =========================================================
# 14. Final summary
# =========================================================
print_header "🎉 FULL PIPELINE FINISHED"

echo "Main final outputs:"
echo "  - $ROOT_DIR/outputs/ptbxl/step3_4_ptxbl_report_llm_label.jsonl"
echo "  - $ROOT_DIR/outputs/ptbxl/ptxbl_report_llm_label_with_vlm_quality.jsonl"
echo
echo "Key intermediate outputs:"
echo "  - $ROOT_DIR/data/interim/ptbxl/ptbxl_database_with_machine_flag.csv"
echo "  - $ROOT_DIR/data/interim/ptbxl/ptbxl_human_report_empty_schema.jsonl"
echo "  - $ROOT_DIR/outputs/ptbxl/step2_ptbxl_human_report_filled.jsonl"
echo "  - $ROOT_DIR/outputs/ptbxl/step2_ptbxl_human_report_error_log.jsonl"
echo "  - $ROOT_DIR/outputs/ptbxl/step3_1_ptbxl_human_report_kept.jsonl"
echo "  - $ROOT_DIR/outputs/ptbxl/step3_1_ptbxl_human_report_dropped.jsonl"
echo "  - $ROOT_DIR/outputs/ptbxl/step3_2_ptbxl_consensus_unmapped.jsonl"
echo "  - $ROOT_DIR/outputs/ptbxl/step3_2_ptbxl_consensus_unmapped_dropped.jsonl"
echo "  - $ROOT_DIR/outputs/ptbxl/step3_3_ptbxl_consensus_unmapped_terms_filled.jsonl"
echo "  - $ROOT_DIR/outputs/ptbxl/step3_4_ptbxl_report_label_stats.jsonl"
echo "  - $ROOT_DIR/outputs/ptbxl/step4_signal_quality_top5.jsonl"
echo "  - $ROOT_DIR/outputs/ptbxl/step5_vlm_quality_results.jsonl"
echo "  - $ROOT_DIR/outputs/ptbxl/step5_vlm_quality_error.jsonl"
echo "  - $ROOT_DIR/outputs/ptbxl/step6_merge_report_label_with_vlm_quality_stats.jsonl"
echo
echo "✅ All done."