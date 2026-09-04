#!/usr/bin/env bash
# Lab-server evals (no Slurm). Same protocol as the Fir slurm jobs.
#
#   cd /path/to/Rectification_dynamic
#   source /path/to/venv/bin/activate
#   export HF_HOME=$HOME/hf-cache
#
#   # Serial (safe, default):
#   bash quick_start/run_eval_lab.sh c1
#   bash quick_start/run_eval_lab.sh c235
#   bash quick_start/run_eval_lab.sh ladder
#
#   # Parallel (faster, needs enough GPUs — one per model):
#   GPUS=0,1,2,3,4,5,6,7 bash quick_start/run_eval_lab.sh c1
#   GPUS=0,1,2,3,4,5,6,7 bash quick_start/run_eval_lab.sh c235
#   GPUS=0,1,2,3,4,5,6,7 bash quick_start/run_eval_lab.sh ladder
#
# GPUS: comma-separated visible GPU ids to assign to models in order.
#       If fewer GPUs than models, remaining models wait (serial fallback).
#       Leave unset to run fully serial (original behaviour).

set -euo pipefail

WHICH="${1:?usage: run_eval_lab.sh c1|c235|ladder}"
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

export HF_HOME="${HF_HOME:-$HOME/hf-cache}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/hub}"
export VLLM_USE_V1="${VLLM_USE_V1:-0}"
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"
export TOKENIZERS_PARALLELISM=true
export PYTHONNOUSERSITE=1

HF_BASE="${HF_BASE:-$HF_HUB_CACHE/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306}"
if [[ ! -d "$HF_BASE" ]]; then
  snap="$(ls -d "$HF_HUB_CACHE"/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/* 2>/dev/null | head -1 || true)"
  HF_BASE="${snap:-$HF_BASE}"
fi
CKPT_ROOT="${CKPT_ROOT:-$REPO_ROOT/checkpoints/PAG-SEC}"
DATA="${DATA:-$REPO_ROOT/datasets/math7500.parquet}"
# math500 eval uses a different parquet
if [[ "$WHICH" == "math500" ]]; then
  DATA="$REPO_ROOT/datasets/math500.parquet"
fi

case "$WHICH" in
  c1)
    COHORT="$REPO_ROOT/tools/c1_cohort_step100.json"
    OUT_DIR="$REPO_ROOT/results/c1_g"
    K="${K:-8}"
    TAGS=(base dynamic_100 dynamic_400 uniform_100 uniform_400)
    ;;
  c235)
    COHORT="$REPO_ROOT/tools/c235_cohort_step100.json"
    OUT_DIR="$REPO_ROOT/results/c235_g"
    K="${K:-8}"
    TAGS=(base dynamic_100 dynamic_400 uniform_100 uniform_400)
    ;;
  ladder)
    COHORT="$REPO_ROOT/tools/train_all_cohort.json"
    OUT_DIR="$REPO_ROOT/results/uniform_g_ladder"
    K="${K:-4}"
    TAGS=(base uniform_100 uniform_200 uniform_300 uniform_400)
    ;;
  math500)
    COHORT="$REPO_ROOT/tools/math500_cohort.json"
    OUT_DIR="$REPO_ROOT/results/math500_g"
    K="${K:-8}"
    TAGS=(base dynamic_100 dynamic_400 uniform_100 uniform_400)
    ;;
  *)
    echo "unknown $WHICH" >&2
    exit 1
    ;;
esac

mkdir -p "$OUT_DIR" "$REPO_ROOT/logs"
test -f "$COHORT"
test -d "$HF_BASE"

model_for() {
  case "$1" in
    base)        echo "$HF_BASE" ;;
    dynamic_100) echo "$CKPT_ROOT/qwen1p5b_sec_dynamic_fir/global_step_100/actor_hf" ;;
    dynamic_400) echo "$CKPT_ROOT/qwen1p5b_sec_dynamic_fir/global_step_400/actor_hf" ;;
    uniform_100) echo "$CKPT_ROOT/qwen1p5b_uniform_disc/global_step_100/actor_hf" ;;
    uniform_200) echo "$CKPT_ROOT/qwen1p5b_uniform_disc/global_step_200/actor_hf" ;;
    uniform_300) echo "$CKPT_ROOT/qwen1p5b_uniform_disc/global_step_300/actor_hf" ;;
    uniform_400) echo "$CKPT_ROOT/qwen1p5b_uniform_disc/global_step_400/actor_hf" ;;
    *) echo "bad tag $1" >&2; return 1 ;;
  esac
}

# Parse GPUS env var into an array (e.g. "0,1,2,3" -> (0 1 2 3))
IFS=',' read -r -a GPU_LIST <<< "${GPUS:-}"

run_one() {
  local tag="$1" gpu="$2"
  local model out
  model="$(model_for "$tag")"
  out="$OUT_DIR/${tag}.json"
  if [[ -f "$out" ]]; then
    echo "[lab-eval] skip $out"
    return 0
  fi
  if [[ ! -f "$model/config.json" ]]; then
    echo "[lab-eval] missing $model  (merge actor_hf first)" >&2
    return 1
  fi
  local log="$REPO_ROOT/logs/lab-eval-${WHICH}-${tag}.log"
  if [[ -n "$gpu" ]]; then
    echo "[lab-eval] $tag  gpu=$gpu  (log: $log)"
    CUDA_VISIBLE_DEVICES="$gpu" python "$REPO_ROOT/tools/eval_c4_g_contrast.py" \
      --model "$model" --tokenizer "$model" \
      --cohort "$COHORT" --tag "$tag" --output "$out" \
      --K "$K" --temperature 1.0 --top_k 10000 --top_p 1.0 \
      --max_tokens 2028 --gpu_util "${GPU_UTIL:-0.85}" \
      --data "$DATA" \
      > "$log" 2>&1
    echo "[lab-eval] $tag done  $(tail -1 "$log")"
  else
    echo "[lab-eval] $tag  (serial)"
    python "$REPO_ROOT/tools/eval_c4_g_contrast.py" \
      --model "$model" --tokenizer "$model" \
      --cohort "$COHORT" --tag "$tag" --output "$out" \
      --K "$K" --temperature 1.0 --top_k 10000 --top_p 1.0 \
      --max_tokens 2028 --gpu_util "${GPU_UTIL:-0.85}" \
      --data "$DATA"
  fi
}

if [[ ${#GPU_LIST[@]} -gt 0 ]]; then
  # ── parallel mode: each tag gets its own GPU ──────────────────────────
  echo "[lab-eval] parallel mode  gpus=(${GPU_LIST[*]})  tags=(${TAGS[*]})"
  pids=()
  for i in "${!TAGS[@]}"; do
    gpu="${GPU_LIST[$i]:-}"           # empty string if fewer GPUs than tags
    if [[ -n "$gpu" ]]; then
      run_one "${TAGS[$i]}" "$gpu" &
      pids+=($!)
    else
      # fallback: run remaining tags serially after parallel batch finishes
      # reuse the last gpu in the list so we don't fall back to GPU 0
      wait "${pids[@]}" 2>/dev/null || true
      pids=()
      run_one "${TAGS[$i]}" "${GPU_LIST[-1]}"
    fi
  done
  # wait for any still-running background jobs
  fail=0
  for pid in "${pids[@]}"; do
    wait "$pid" || fail=1
  done
  if [[ "$fail" -ne 0 ]]; then
    echo "[lab-eval] one or more parallel jobs failed" >&2
    exit 1
  fi
else
  # ── serial mode (original behaviour) ──────────────────────────────────
  for tag in "${TAGS[@]}"; do
    run_one "$tag" ""
  done
fi

if [[ "$WHICH" == "ladder" ]]; then
  python "$REPO_ROOT/tools/build_hindsight_stages.py" --eval-dir "$OUT_DIR"
elif [[ "$WHICH" == "math500" ]]; then
  python "$REPO_ROOT/tools/summarize_math500.py" --eval-dir "$OUT_DIR"
elif [[ "$WHICH" == "c1" || "$WHICH" == "c235" ]]; then
  python "$REPO_ROOT/tools/eval_c4_g_contrast.py" --summarize "$OUT_DIR" --cohort "$COHORT" || true
fi
echo "[lab-eval] $WHICH done"
