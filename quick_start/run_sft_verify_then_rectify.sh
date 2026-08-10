#!/usr/bin/env bash
# Sequential SFT: verify warmup → rectify (init from verify ckpt).
# Override VERIFY_STEP to pick a specific global_step_* under the verify run dir.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

N_GPUS="${N_GPUS:-8}"
VERIFY_EXP="${VERIFY_EXP:-qwen_pag_sft_verify}"
RECTIFY_EXP="${RECTIFY_EXP:-qwen_pag_sft_rectify}"

run_sft() {
  # Forward selected vars into the child without brittle ${VAR:+...} continuations.
  local -a cmd=(env "SPLIT=$1" "N_GPUS=$N_GPUS" "EXPERIMENT_NAME=$2")
  if [[ -n "${MODEL_PATH:-}" ]]; then
    cmd+=("MODEL_PATH=$MODEL_PATH")
  fi
  # Optional knobs (inherit from caller if set)
  for v in USE_WANDB TRAIN_BATCH_SIZE MICRO_BATCH_SIZE_PER_GPU MAX_LENGTH LR EPOCHS \
           PROJECT_NAME CKPT_PATH MASTER_PORT CUDA_VISIBLE_DEVICES \
           EVAL_EVERY EVAL_N_PROBLEMS EVAL_DATA_PATH EVAL_MAX_NEW_TOKENS EVAL_TEMPERATURE; do
    if [[ -n "${!v:-}" ]]; then
      cmd+=("$v=${!v}")
    fi
  done
  cmd+=(bash "$REPO_ROOT/quick_start/run_sft_pag_multiturn.sh")
  "${cmd[@]}"
}

echo "=== stage 1: verify SFT ==="
run_sft verify "$VERIFY_EXP"

VERIFY_DIR="$REPO_ROOT/checkpoints/sft/$VERIFY_EXP"
if [[ -n "${VERIFY_STEP:-}" ]]; then
  INIT_CKPT="$VERIFY_DIR/global_step_${VERIFY_STEP}"
else
  INIT_CKPT="$(ls -d "$VERIFY_DIR"/global_step_* 2>/dev/null | sort -V | tail -1 || true)"
fi
if [[ -z "${INIT_CKPT}" || ! -d "${INIT_CKPT}" ]]; then
  echo "No verify checkpoint under $VERIFY_DIR — check stage 1." >&2
  exit 1
fi

echo "=== stage 2: rectify SFT (init=$INIT_CKPT) ==="
MODEL_PATH="$INIT_CKPT" run_sft rectify "$RECTIFY_EXP"

echo "=== sequential SFT finished ==="
