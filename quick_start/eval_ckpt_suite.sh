#!/usr/bin/env bash
# Eval one actor ckpt (or the Instruct base) on MATH-500 / MinervaMATH / AIME24 / AIME25.
# Protocol matches training val: PAG 2-turn, n=VAL_N (default 8), T=0.6, top_p=0.95.
#
# Usage:
#   NAME=sec_dynamic N_GPUS=6 RESUME_PATH=.../global_step_400 bash quick_start/eval_ckpt_suite.sh
#   NAME=instruct    N_GPUS=8 RESUME_MODE=disable bash quick_start/eval_ckpt_suite.sh
set -euo pipefail
set -x

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

export HF_HOME="${HF_HOME:-/data/yuranli/hf-cache}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-/data/yuranli/hf-cache/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-/data/yuranli/hf-cache/hub}"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_USE_V1=0
export TOKENIZERS_PARALLELISM=true
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export PYTHONNOUSERSITE=1
export RAY_TMPDIR="${RAY_TMPDIR:-/data/yuranli/tmp/ray_eval_suite}"

PY="${PY:-/data/yuranli/envs/PAG/bin/python}"

math500="$REPO_ROOT/datasets/math500.parquet"
aime2024="$REPO_ROOT/datasets/aime2024.parquet"
aime2025="$REPO_ROOT/datasets/aime2025.parquet"
minervamath="$REPO_ROOT/datasets/minervamath.parquet"
dapo17k="$REPO_ROOT/datasets/dapo17k.parquet"

NAME="${NAME:-eval}"
N_GPUS="${N_GPUS:-8}"
VAL_N="${VAL_N:-8}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.70}"
RESUME_MODE="${RESUME_MODE:-resume_path}"
RESUME_PATH="${RESUME_PATH:-None}"
MODEL_PATH="${MODEL_PATH:-/data/yuranli/hf-cache/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306}"
PROJECT_NAME="${PROJECT_NAME:-PAG-SEC-eval}"
CKPT_PATH="${CKPT_PATH:-$REPO_ROOT/checkpoints}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-eval_${NAME}_n${VAL_N}}"
LOG_DIR="${LOG_DIR:-$REPO_ROOT/results/eval_suite}"
mkdir -p "$LOG_DIR" "$RAY_TMPDIR"

# train_batch_size * rollout.n must be divisible by n_gpus (config assert only).
if [[ "$N_GPUS" -eq 6 ]]; then
  TRAIN_BS=510
else
  TRAIN_BS=512
fi

VAL_FILES="['$math500','$minervamath','$aime2024','$aime2025']"
VALIDATION_JSON="$LOG_DIR/${NAME}_n${VAL_N}.json"

EXTRA_RESUME=()
if [[ "$RESUME_MODE" == "resume_path" ]]; then
  EXTRA_RESUME+=(trainer.resume_from_path="$RESUME_PATH")
fi

"$PY" -m verl.trainer.main_ppo \
    trainer.resume_mode="$RESUME_MODE" \
    "${EXTRA_RESUME[@]}" \
    trainer.val_before_train=True \
    trainer.val_only=True \
    algorithm.adv_estimator=grpo \
    data.train_files="[$dapo17k]" \
    data.val_files="$VAL_FILES" \
    data.filter_overlong_prompts=True \
    data.train_batch_size=$TRAIN_BS \
    data.max_prompt_length=1024 \
    data.max_response_length=2028 \
    actor_rollout_ref.model.path="$MODEL_PATH" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=16384 \
    actor_rollout_ref.rollout.max_num_batched_tokens=16384 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization="$GPU_MEM_UTIL" \
    actor_rollout_ref.rollout.num_turns=2 \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.rollout_type=pag \
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.rollout.val_kwargs.n=$VAL_N \
    actor_rollout_ref.rollout.val_kwargs.num_turns=2 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.top_k=-1 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.95 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.6 \
    reward_model.policy_rs=True \
    reward_model.rs_coef=1.0 \
    reward_model.feedback_mode=disc \
    algorithm.norm_type=role \
    algorithm.use_kl_in_reward=False \
    actor_rollout_ref.actor.use_kl_loss=False \
    sec.enabled=False \
    curriculum.enabled=False \
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.logger="['console']" \
    trainer.n_gpus_per_node=$N_GPUS \
    trainer.nnodes=1 \
    trainer.total_epochs=1 \
    trainer.default_local_dir=$CKPT_PATH/$PROJECT_NAME/$EXPERIMENT_NAME \
    trainer.log_val_generations=2 \
    trainer.save_validation_results=True \
    trainer.validation_results_path="$VALIDATION_JSON"
