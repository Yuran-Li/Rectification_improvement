#!/usr/bin/env bash
# Multi-turn SFT on S2R→PAG converted data (verify / rectify / mixed).
#
# Prerequisites:
#   python tools/sft_data/convert_s2r_to_pag_multiturn.py --out_dir datasets/sft
#
# Examples:
#   # verify-only warmup (recommended first)
#   SPLIT=verify N_GPUS=4 bash quick_start/run_sft_pag_multiturn.sh
#
#   # rectify-only (optionally init from verify ckpt via MODEL_PATH)
#   SPLIT=rectify MODEL_PATH=$REPO/checkpoints/sft/.../global_step_X \
#     bash quick_start/run_sft_pag_multiturn.sh
#
#   # joint warmup
#   SPLIT=mixed bash quick_start/run_sft_pag_multiturn.sh
#
set -euo pipefail
set -x

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

export HF_HOME="${HF_HOME:-/data/yuranli/hf-cache}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/hub}"
export TOKENIZERS_PARALLELISM=true
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export PYTHONNOUSERSITE=1
# Prefer this repo's verl over any other install
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

SPLIT="${SPLIT:-mixed}"   # verify | rectify | mixed
N_GPUS="${N_GPUS:-8}"
NNODES="${NNODES:-1}"
MASTER_PORT="${MASTER_PORT:-29551}"

DEFAULT_MODEL_SNAP="$HF_HUB_CACHE/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
if [[ -d "$DEFAULT_MODEL_SNAP" ]]; then
  DEFAULT_MODEL="$DEFAULT_MODEL_SNAP"
else
  DEFAULT_MODEL="Qwen/Qwen2.5-1.5B-Instruct"
fi
MODEL_PATH="${MODEL_PATH:-$DEFAULT_MODEL}"

TRAIN_FILE="${TRAIN_FILE:-$REPO_ROOT/datasets/sft/sft_${SPLIT}_train.parquet}"
VAL_FILE="${VAL_FILE:-$REPO_ROOT/datasets/sft/sft_${SPLIT}_val.parquet}"
for f in "$TRAIN_FILE" "$VAL_FILE"; do
  if [[ ! -f "$f" ]]; then
    echo "Missing $f — run: python tools/sft_data/convert_s2r_to_pag_multiturn.py --out_dir datasets/sft" >&2
    exit 1
  fi
done

PROJECT_NAME="${PROJECT_NAME:-Rectification_SFT}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-qwen_pag_sft_${SPLIT}}"
CKPT_PATH="${CKPT_PATH:-$REPO_ROOT/checkpoints/sft}"
mkdir -p "$CKPT_PATH"

# Global batch must be divisible by N_GPUS and by micro_batch_size_per_gpu
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-64}"
MICRO_BATCH_SIZE_PER_GPU="${MICRO_BATCH_SIZE_PER_GPU:-1}"
MAX_LENGTH="${MAX_LENGTH:-8192}"   # rectify trajectories can be long
LR="${LR:-5e-6}"
EPOCHS="${EPOCHS:-3}"
TRUNCATION="${TRUNCATION:-right}"  # avoid hard crash on rare long examples

# Periodic PAG generative eval on MATH500 subset (0 = disabled)
EVAL_EVERY="${EVAL_EVERY:-20}"
EVAL_N_PROBLEMS="${EVAL_N_PROBLEMS:-32}"
EVAL_DATA_PATH="${EVAL_DATA_PATH:-$REPO_ROOT/datasets/math500.parquet}"
EVAL_MAX_NEW_TOKENS="${EVAL_MAX_NEW_TOKENS:-1024}"
EVAL_TEMPERATURE="${EVAL_TEMPERATURE:-0.0}"

# Auto-align global batch to N_GPUS (e.g. 64 with 6 GPUs -> 60)
if (( TRAIN_BATCH_SIZE % N_GPUS != 0 )); then
  ALIGNED=$(( TRAIN_BATCH_SIZE / N_GPUS * N_GPUS ))
  if (( ALIGNED < N_GPUS )); then
    ALIGNED=$N_GPUS
  fi
  echo "[SFT] WARN: TRAIN_BATCH_SIZE=$TRAIN_BATCH_SIZE not divisible by N_GPUS=$N_GPUS; using $ALIGNED"
  TRAIN_BATCH_SIZE=$ALIGNED
fi
if (( (TRAIN_BATCH_SIZE / N_GPUS) % MICRO_BATCH_SIZE_PER_GPU != 0 )); then
  echo "[SFT] ERROR: per-GPU batch $((TRAIN_BATCH_SIZE / N_GPUS)) not divisible by MICRO_BATCH_SIZE_PER_GPU=$MICRO_BATCH_SIZE_PER_GPU" >&2
  exit 1
fi

if [[ "${USE_WANDB:-0}" == "1" ]]; then
  LOGGER="['console','wandb']"
else
  LOGGER="['console']"
fi

echo "[SFT] split=$SPLIT model=$MODEL_PATH gpus=$N_GPUS batch=$TRAIN_BATCH_SIZE micro=$MICRO_BATCH_SIZE_PER_GPU maxlen=$MAX_LENGTH eval_every=$EVAL_EVERY n_eval=$EVAL_N_PROBLEMS"

# Visible GPU count must be >= N_GPUS (otherwise: CUDA invalid device ordinal)
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  N_VISIBLE=$(awk -F',' '{print NF}' <<<"$CUDA_VISIBLE_DEVICES")
else
  N_VISIBLE=$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')
fi
if [[ -z "$N_VISIBLE" || "$N_VISIBLE" -lt "$N_GPUS" ]]; then
  echo "[SFT] ERROR: visible GPUs=$N_VISIBLE < N_GPUS=$N_GPUS (CUDA_VISIBLE_DEVICES='${CUDA_VISIBLE_DEVICES:-}')" >&2
  exit 1
fi

torchrun \
  --standalone \
  --nnodes="$NNODES" \
  --nproc_per_node="$N_GPUS" \
  --master_port="$MASTER_PORT" \
  -m verl.trainer.fsdp_sft_trainer \
  data.train_files="$TRAIN_FILE" \
  data.val_files="$VAL_FILE" \
  data.multiturn.enable=true \
  data.multiturn.messages_key=messages \
  data.train_batch_size="$TRAIN_BATCH_SIZE" \
  data.micro_batch_size_per_gpu="$MICRO_BATCH_SIZE_PER_GPU" \
  data.max_length="$MAX_LENGTH" \
  data.truncation="$TRUNCATION" \
  data.balance_dp_token=False \
  model.partial_pretrain="$MODEL_PATH" \
  model.enable_gradient_checkpointing=True \
  model.trust_remote_code=True \
  model.fsdp_config.cpu_offload=False \
  optim.lr="$LR" \
  optim.warmup_steps_ratio=0.05 \
  trainer.default_local_dir="$CKPT_PATH/$EXPERIMENT_NAME" \
  trainer.project_name="$PROJECT_NAME" \
  trainer.experiment_name="$EXPERIMENT_NAME" \
  trainer.total_epochs="$EPOCHS" \
  trainer.logger="$LOGGER" \
  trainer.seed=42 \
  trainer.eval_every="$EVAL_EVERY" \
  trainer.eval_n_problems="$EVAL_N_PROBLEMS" \
  trainer.eval_data_path="$EVAL_DATA_PATH" \
  trainer.eval_max_new_tokens="$EVAL_MAX_NEW_TOKENS" \
  trainer.eval_temperature="$EVAL_TEMPERATURE"

echo "[SFT] done. checkpoints under $CKPT_PATH/$EXPERIMENT_NAME"
