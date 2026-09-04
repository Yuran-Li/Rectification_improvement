#!/usr/bin/env bash
# Sequential PAG val_only for Instruct + 4 step-400 ckpts.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"
LOG_DIR="$REPO_ROOT/results/eval_suite"
mkdir -p "$LOG_DIR"
MASTER_LOG="$LOG_DIR/suite_$(date +%Y%m%d_%H%M%S).log"

run_one() {
  local name="$1"
  shift
  echo "========== $(date) START $name ==========" | tee -a "$MASTER_LOG"
  if env "$@" bash "$REPO_ROOT/quick_start/eval_ckpt_suite.sh" \
      >"$LOG_DIR/${name}.log" 2>&1; then
    echo "========== $(date) OK $name ==========" | tee -a "$MASTER_LOG"
  else
    echo "========== $(date) FAIL $name (exit $?) ==========" | tee -a "$MASTER_LOG"
    return 1
  fi
  # drop leftover Ray so the next job can claim all GPUs
  /data/yuranli/envs/PAG/bin/ray stop --force >/dev/null 2>&1 || true
  sleep 5
}

ROOT_CKPT="$REPO_ROOT/checkpoints/PAG-SEC"

run_one instruct \
  NAME=instruct N_GPUS=8 RESUME_MODE=disable

run_one sec_dynamic \
  NAME=sec_dynamic N_GPUS=6 RESUME_MODE=resume_path \
  RESUME_PATH="$ROOT_CKPT/qwen1p5b_sec_dynamic/global_step_400"

run_one sec_dynamic_fir \
  NAME=sec_dynamic_fir N_GPUS=8 RESUME_MODE=resume_path \
  RESUME_PATH="$ROOT_CKPT/qwen1p5b_sec_dynamic_fir/global_step_400"

run_one sec_fixed \
  NAME=sec_fixed N_GPUS=8 RESUME_MODE=resume_path \
  RESUME_PATH="$ROOT_CKPT/qwen1p5b_sec_fixed/global_step_400"

run_one uniform_disc \
  NAME=uniform_disc N_GPUS=8 RESUME_MODE=resume_path \
  RESUME_PATH="$ROOT_CKPT/qwen1p5b_uniform_disc/global_step_400"

echo "========== $(date) ALL DONE ==========" | tee -a "$MASTER_LOG"
