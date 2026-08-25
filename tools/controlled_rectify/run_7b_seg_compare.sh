#!/usr/bin/env bash
# Compare SCOPE (\\n\\n) vs STRIDE (<step>) segmentation on Qwen2.5-7B-Instruct's own wrong pool.
# Focus: prefix_rewrite + localization_analysis vs freeform.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
CR="$ROOT/tools/controlled_rectify"
PY="${PAG_PY:-/data/yuranli/envs/PAG/bin/python}"
Q7="${Q7:-/data/yuranli/hf-cache/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28}"
MATH500="${MATH500:-$ROOT/datasets/math500.parquet}"
TP="${TP:-1}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.75}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-64}"
export VLLM_USE_V1="${VLLM_USE_V1:-0}"
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"
export PYTHONNOUSERSITE=1

POOL_SCOPE="${POOL_SCOPE:-$CR/data/fixed_wrong_qwen7b_instruct_scope.jsonl}"
POOL_STRIDE="${POOL_STRIDE:-$CR/data/fixed_wrong_qwen7b_instruct_stride.jsonl}"
TEACH_SCOPE="${TEACH_SCOPE:-$CR/data/fixed_wrong_qwen7b_instruct_scope_teacher.jsonl}"
TEACH_STRIDE="${TEACH_STRIDE:-$CR/data/fixed_wrong_qwen7b_instruct_stride_teacher.jsonl}"
OUT_SCOPE="${OUT_SCOPE:-$CR/results/causal_rectify_qwen7b_scope.jsonl}"
OUT_STRIDE="${OUT_STRIDE:-$CR/results/causal_rectify_qwen7b_stride.jsonl}"

# Full 2D W2C matrix (same feedback axis as pag_ppo400 / 1.5B Instruct tables)
FB="${FB:-regenerate wrong_only localization localization_analysis localization_analysis_plan freeform}"
EDITS="${EDITS:-full_regen prefix_rewrite}"
STAGE="${1:-all}"

# Teacher: gpt if OPENAI_API_KEY else local 7B
if [[ -n "${OPENAI_API_KEY:-}" ]]; then
  CAUSAL_BACKEND="${CAUSAL_BACKEND:-gpt}"
  TEACHER_MODEL="${TEACHER_MODEL:-${GPT_CRITIQUE_MODEL:-gpt-5o}}"
else
  CAUSAL_BACKEND="${CAUSAL_BACKEND:-local}"
  TEACHER_MODEL="${TEACHER_MODEL:-$Q7}"
  echo "[warn] OPENAI_API_KEY unset -> teacher backend=local model=$TEACHER_MODEL"
fi

build_pools() {
  echo "[$(date)] build SCOPE pool (free + blank lines) -> $POOL_SCOPE"
  CUDA_VISIBLE_DEVICES="${CUDA_SCOPE:-0}" "$PY" "$CR/build_prerl_pool.py" \
    --model_path "$Q7" --parquet "$MATH500" --out "$POOL_SCOPE" \
    --gen_format free --tensor_parallel_size "$TP" \
    --gpu_memory_utilization "$GPU_MEM_UTIL" --max_num_seqs "$MAX_NUM_SEQS"

  echo "[$(date)] build STRIDE pool (<step> tags) -> $POOL_STRIDE"
  CUDA_VISIBLE_DEVICES="${CUDA_STRIDE:-1}" "$PY" "$CR/build_prerl_pool.py" \
    --model_path "$Q7" --parquet "$MATH500" --out "$POOL_STRIDE" \
    --gen_format stride_tags --tensor_parallel_size "$TP" \
    --gpu_memory_utilization "$GPU_MEM_UTIL" --max_num_seqs "$MAX_NUM_SEQS"
}

teachers() {
  echo "[$(date)] teacher SCOPE segment_method=scope_nn backend=$CAUSAL_BACKEND"
  # shellcheck disable=SC2086
  CUDA_VISIBLE_DEVICES="${CUDA_TEACH:-0}" "$PY" "$CR/generate_structured_feedback.py" \
    --input "$POOL_SCOPE" --output "$TEACH_SCOPE" \
    --backend "$CAUSAL_BACKEND" --model "$TEACHER_MODEL" \
    --segment_method scope_nn \
    --tensor_parallel_size "$TP" \
    --gpu_memory_utilization "$GPU_MEM_UTIL" --max_num_seqs "$MAX_NUM_SEQS"

  echo "[$(date)] teacher STRIDE segment_method=stride_tags backend=$CAUSAL_BACKEND"
  CUDA_VISIBLE_DEVICES="${CUDA_TEACH:-0}" "$PY" "$CR/generate_structured_feedback.py" \
    --input "$POOL_STRIDE" --output "$TEACH_STRIDE" \
    --backend "$CAUSAL_BACKEND" --model "$TEACHER_MODEL" \
    --segment_method stride_tags \
    --tensor_parallel_size "$TP" \
    --gpu_memory_utilization "$GPU_MEM_UTIL" --max_num_seqs "$MAX_NUM_SEQS"
}

evals() {
  echo "[$(date)] eval SCOPE rectifier=7B"
  # shellcheck disable=SC2086
  CUDA_VISIBLE_DEVICES="${CUDA_EVAL:-0}" "$PY" "$CR/causal_rectify_eval.py" \
    --model_path "$Q7" --data "$TEACH_SCOPE" --output "$OUT_SCOPE" \
    --feedbacks $FB --edit_modes $EDITS \
    --n_samples 1 --temperature 0.0 \
    --tensor_parallel_size "$TP" \
    --gpu_memory_utilization "$GPU_MEM_UTIL" --max_num_seqs "$MAX_NUM_SEQS"

  echo "[$(date)] eval STRIDE rectifier=7B"
  CUDA_VISIBLE_DEVICES="${CUDA_EVAL:-0}" "$PY" "$CR/causal_rectify_eval.py" \
    --model_path "$Q7" --data "$TEACH_STRIDE" --output "$OUT_STRIDE" \
    --feedbacks $FB --edit_modes $EDITS \
    --n_samples 1 --temperature 0.0 \
    --tensor_parallel_size "$TP" \
    --gpu_memory_utilization "$GPU_MEM_UTIL" --max_num_seqs "$MAX_NUM_SEQS"

  "$PY" "$CR/aggregate_causal_rectify.py" --input "$OUT_SCOPE" --out_prefix "${OUT_SCOPE%.jsonl}"
  "$PY" "$CR/aggregate_causal_rectify.py" --input "$OUT_STRIDE" --out_prefix "${OUT_STRIDE%.jsonl}"
  "$PY" - <<'PY'
import json
from pathlib import Path
CR = Path("/data/yuranli/LLM/2026.04/github_references/Policy-As-GenVerifier/tools/controlled_rectify/results")
FBS = [
    ("regenerate", "Regenerate（no feedback）"),
    ("wrong_only", "Wrong only"),
    ("localization", "Localization"),
    ("localization_analysis", "Loc.+Analysis"),
    ("localization_analysis_plan", "Loc.+Analysis+Plan"),
    ("freeform", "Freeform why/how"),
]
for tag in ["qwen7b_scope", "qwen7b_stride"]:
    p = CR / f"causal_rectify_{tag}.summary.json"
    if not p.exists():
        print(tag, "missing"); continue
    s = json.loads(p.read_text()); m = s["metrics"]
    print(f"\n=== {tag} n={s['n_rows']}  W2C@1 (%) ===")
    print(f"{'Feedback':28s} {'Full regenerate':>16s} {'Prefix rewrite':>16s}")
    for fb, label in FBS:
        fr = m.get(f"{fb}__full_regen", {}).get("W2C@1_pct")
        pr = m.get(f"{fb}__prefix_rewrite", {}).get("W2C@1_pct")
        print(f"{label:28s} {fr!s:>16} {pr!s:>16}")
    loc_p = m.get("localization_analysis__prefix_rewrite", {}).get("W2C@1_pct")
    free_p = m.get("freeform__prefix_rewrite", {}).get("W2C@1_pct")
    if loc_p is not None and free_p is not None:
        print(f"  >> prefix: Loc.+Analysis - Freeform = {loc_p - free_p:+.1f}")
PY
}

case "$STAGE" in
  pools) build_pools ;;
  teachers) teachers ;;
  evals) evals ;;
  all)
    build_pools
    teachers
    evals
    ;;
  *)
    echo "Usage: $0 {pools|teachers|evals|all}"
    exit 1
    ;;
esac
