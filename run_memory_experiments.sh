#!/usr/bin/env bash
set -euo pipefail

# MedGym memory experiments for self-evolving multimodal memory system.
#
# Assumptions:
# - Doctor / Patient / Judge vLLM servers are already running.
# - Retrieval server is already running if you want retrieval-enabled runs.
# - This script only launches MedGym rollouts with different memory configs.
#
# Usage:
#   bash run_memory_experiments.sh
#
# Optional env overrides:
#   CASE_DIR=/oral_llm/xiweidai/med_env/bench
#   MODEL_NAME=doctor_agent
#   TOKENIZER_PATH=/oral_llm/xiweidai/med_env/models/Qwen3-VL-8B-Instruct
#   BASE_URL=http://127.0.0.1:30000/v1
#   JUDGE_MODEL=judge_agent
#   JUDGE_BASE_URL=http://127.0.0.1:30002/v1
#   EXPERIMENT_SUITE=full
#   MAX_CASES=0
#   REPEAT_K=1
#   NO_CXR=0
#   MEMORY_LLM_MODEL=
#   MEMORY_LLM_BASE_URL=
#   MEMORY_LLM_API_KEY=

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${SCRIPT_DIR}"

cd "${PROJECT_ROOT}"

CASE_DIR="${CASE_DIR:-/oral_llm/xiweidai/med_env/bench}"
MODEL_NAME="${MODEL_NAME:-doctor_agent}"
TOKENIZER_PATH="${TOKENIZER_PATH:-/oral_llm/xiweidai/med_env/models/Qwen3-VL-8B-Instruct}"
BASE_URL="${BASE_URL:-http://127.0.0.1:30000/v1}"
API_KEY="${API_KEY:-None}"
JUDGE_MODEL="${JUDGE_MODEL:-judge_agent}"
JUDGE_BASE_URL="${JUDGE_BASE_URL:-http://127.0.0.1:30002/v1}"
JUDGE_API_KEY="${JUDGE_API_KEY:-None}"
EXPERIMENT_SUITE="${EXPERIMENT_SUITE:-full}"
MAX_CASES="${MAX_CASES:-0}"
REPEAT_K="${REPEAT_K:-1}"
NO_CXR="${NO_CXR:-0}"
N_PARALLEL_AGENTS="${N_PARALLEL_AGENTS:-1}"
PYTHON="${PYTHON:-python}"
LOG_ROOT="${LOG_ROOT:-${PROJECT_ROOT}/logs/medmemory_runs}"
MEMORY_LLM_MODEL="${MEMORY_LLM_MODEL:-}"
MEMORY_LLM_BASE_URL="${MEMORY_LLM_BASE_URL:-}"
MEMORY_LLM_API_KEY="${MEMORY_LLM_API_KEY:-}"

mkdir -p "${LOG_ROOT}"

DATA_ARGS=()
if [[ "${NO_CXR}" == "1" ]]; then
  DATA_ARGS+=(--no_cxr)
fi

JUDGE_ARGS=(--judge_model "${JUDGE_MODEL}" --judge_base_url "${JUDGE_BASE_URL}" --judge_api_key "${JUDGE_API_KEY}")

BASE_COMMON_ARGS=(
  --model "${MODEL_NAME}"
  --tokenizer_path "${TOKENIZER_PATH}"
  --base_url "${BASE_URL}"
  --api_key "${API_KEY}"
  --case_dir "${CASE_DIR}"
  --max_cases "${MAX_CASES}"
  --repeat_k "${REPEAT_K}"
  --n_parallel_agents "${N_PARALLEL_AGENTS}"
  --temperature 0.6
  --top_p 0.95
  --max_prompt_length 8192
  --max_response_length 16384
  --parser_name qwen
  "${DATA_ARGS[@]}"
  "${JUDGE_ARGS[@]}"
)

run_experiment() {
  local name="$1"
  shift
  echo "[INFO] Running experiment: ${name}"
  local out_dir="${LOG_ROOT}/${name}"
  mkdir -p "${out_dir}"
  local stdout_log="${out_dir}/stdout.log"
  local stderr_log="${out_dir}/stderr.log"

  {
    echo "[INFO] Timestamp: $(date -Iseconds)"
    echo "[INFO] Experiment: ${name}"
    echo "[INFO] Args: $*"
  } | tee "${stdout_log}"

  if [[ -n "${MEMORY_LLM_MODEL}" ]]; then
    export MEMORY_LLM_MODEL
  fi
  if [[ -n "${MEMORY_LLM_BASE_URL}" ]]; then
    export MEMORY_LLM_BASE_URL
  fi
  if [[ -n "${MEMORY_LLM_API_KEY}" ]]; then
    export MEMORY_LLM_API_KEY
  fi

  ${PYTHON} run_med_with_tool.py \
    "${BASE_COMMON_ARGS[@]}" \
    "$@" \
    > >(tee -a "${stdout_log}") \
    2> >(tee -a "${stderr_log}" >&2)
}

case "${EXPERIMENT_SUITE}" in
  quick)
    run_experiment "rule_baseline" \
      --query_builder_mode rule \
      --applicability_mode rule \
      --experience_extraction_mode rule \
      --experience_merge_mode rule \
      --memory_top_k 5
    ;;
  main)
    run_experiment "llm_query_hybrid_app" \
      --query_builder_mode llm \
      --applicability_mode hybrid \
      --experience_extraction_mode rule \
      --experience_merge_mode rule \
      --memory_top_k 5
    ;;
  full)
    run_experiment "disable_memory" \
      --disable_memory \
      --query_builder_mode rule \
      --applicability_mode rule \
      --experience_extraction_mode rule \
      --experience_merge_mode rule \
      --memory_top_k 5

    run_experiment "case_memory_only" \
      --query_builder_mode rule \
      --applicability_mode rule \
      --experience_extraction_mode rule \
      --experience_merge_mode rule \
      --memory_top_k 5 \
      --disable_experience_memory \
      --disable_skill_memory \
      --disable_knowledge_memory

    run_experiment "rule_retrieval" \
      --query_builder_mode rule \
      --applicability_mode rule \
      --experience_extraction_mode rule \
      --experience_merge_mode rule \
      --memory_top_k 5

    run_experiment "llm_query_builder" \
      --query_builder_mode llm \
      --applicability_mode rule \
      --experience_extraction_mode rule \
      --experience_merge_mode rule \
      --memory_top_k 5

    run_experiment "hybrid_applicability" \
      --query_builder_mode llm \
      --applicability_mode hybrid \
      --experience_extraction_mode rule \
      --experience_merge_mode rule \
      --memory_top_k 5

    run_experiment "full_memory_pipeline" \
      --query_builder_mode llm \
      --applicability_mode hybrid \
      --experience_extraction_mode llm \
      --experience_merge_mode llm \
      --memory_top_k 5 \
      --log_memory_trace
    ;;
  *)
    echo "[ERROR] Unknown EXPERIMENT_SUITE=${EXPERIMENT_SUITE}. Use quick|main|full."
    exit 1
    ;;
esac

echo "[INFO] Done. Logs saved under ${LOG_ROOT}."
