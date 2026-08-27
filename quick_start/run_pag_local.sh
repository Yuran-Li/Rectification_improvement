#!/usr/bin/env bash
# Local / cluster PAG reproduction (FSDP + vLLM).
# Single node (e.g. 8 GPUs):  N_GPUS=8 NNODES=1 bash quick_start/run_pag_local.sh
# Narval 2x4 A100:           started via quick_start/train_pag_narval.slurm
set -euo pipefail
set -x

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

# Prefer local HF cache / model snapshot
export HF_HOME="${HF_HOME:-${SCRATCH:-/scratch/$USER}/hf-cache}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/hub}"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
# vLLM 0.8 defaults to V1 engine; its profile_run hits
# "Could not infer dtype of numpy.int64" with this stack. Use V0.
export VLLM_USE_V1=0
export TOKENIZERS_PARALLELISM=true
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
# Avoid ~/.local site-packages polluting this env
export PYTHONNOUSERSITE=1

math500="$REPO_ROOT/datasets/math500.parquet"
math7500="$REPO_ROOT/datasets/math7500.parquet"
dapo17k="$REPO_ROOT/datasets/dapo17k.parquet"

PROJECT_NAME='PAG-critique-utility'
CKPT_PATH="${CKPT_PATH:-$REPO_ROOT/checkpoints}"
# Hub id or local snapshot path. Override MODEL_PATH for a local cache.
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-7B-Instruct}"

# 1.5B → MATH 7.5k; 7B → DAPO 17k. Override with TRAIN_DATASET=/path/to.parquet
_model_lc="$(printf '%s' "$MODEL_PATH" | tr '[:upper:]' '[:lower:]')"
if [[ -n "${TRAIN_DATASET:-}" ]]; then
  training_dataset="$TRAIN_DATASET"
elif [[ "$_model_lc" =~ (^|[^0-9])7b([^0-9]|$) ]]; then
  training_dataset="$dapo17k"
elif [[ "$_model_lc" =~ (^|[^0-9])1\.5b([^0-9]|$) ]]; then
  training_dataset="$math7500"
else
  echo "Cannot infer train set from MODEL_PATH=$MODEL_PATH (expected 1.5B or 7B). Set TRAIN_DATASET." >&2
  exit 1
fi
echo "[run_pag_local] MODEL_PATH=$MODEL_PATH  training_dataset=$training_dataset"

EXPERIMENT_NAME="${EXPERIMENT_NAME:-qwen7b_pag}"
n=4
rollout_type=pag
num_turns=2
policy_rs=True
rs_coef=1.0
norm_type=role
split_verify_reward=True
# generic | regen | delta | acc | disc
#   generic: R_self - R_generic (needs y_generic fork)
#   regen:   Δ_self * [1 + λ (1 - p_regen)]
#   delta:   Δ_self = acc_t2 - acc_t1
#   acc:     acc_t2 - λ_cw · 1[C→W]
#   disc:    R_use = 0 (original PAG: verify = genrm_score only)
# Compat: LAMBDA_REGEN>0 without FEEDBACK_MODE still means delta.
if [[ -n "${FEEDBACK_MODE:-}" ]]; then
  feedback_mode="$FEEDBACK_MODE"
else
  case "${LAMBDA_REGEN:-0}" in
    0|0.0|0.00) feedback_mode=generic ;;
    *) feedback_mode=delta ;;
  esac
fi
case "$feedback_mode" in
  generic|critique|r_critique) feedback_mode=generic ;;
  regen) feedback_mode=regen ;;
  delta|base|delta_self) feedback_mode=delta ;;
  acc|acc_t2|acc_cw) feedback_mode=acc ;;
  disc|none|pag|disc_only) feedback_mode=disc ;;
  *) echo "FEEDBACK_MODE must be generic|regen|delta|acc|disc (got: $feedback_mode)" >&2; exit 1 ;;
esac
lambda_regen="${LAMBDA_REGEN:-1.0}"
lambda_cw="${LAMBDA_CW:-0.2}"

# Curriculum sampling (generation-frontier).
curriculum_enabled="${CURRICULUM_ENABLED:-false}"
curriculum_epsilon="${CURRICULUM_EPSILON:-0.3}"

if [[ -n "${GENERIC_COUNTERFACTUAL:-}" ]]; then
  generic_counterfactual="$GENERIC_COUNTERFACTUAL"
elif [[ "$feedback_mode" == "generic" ]]; then
  generic_counterfactual=True
else
  generic_counterfactual=False
fi

# Default: console only (set USE_WANDB=1 to enable wandb)
if [[ "${USE_WANDB:-0}" == "1" ]]; then
  LOGGER="['console','wandb']"
else
  LOGGER="['console']"
fi

# World size: N_GPUS per node × NNODES (Narval: often 4 GPUs/node × 2 nodes)
N_GPUS="${N_GPUS:-8}"
NNODES="${NNODES:-1}"

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=gae \
    data.train_files=[$training_dataset] \
    data.val_files="['$math500']" \
    data.filter_overlong_prompts=True \
    data.train_batch_size=512 \
    data.max_prompt_length=1024 \
    data.max_response_length=2048 \
    "actor_rollout_ref.model.path='${MODEL_PATH}'" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=16384 \
    actor_rollout_ref.rollout.max_num_batched_tokens=16384 \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0 \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.clip_ratio_high=0.28 \
    actor_rollout_ref.actor.clip_ratio_low=0.2 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=${GPU_MEM_UTIL:-0.5} \
    actor_rollout_ref.rollout.n=$n \
    actor_rollout_ref.rollout.top_k=10000 \
    actor_rollout_ref.rollout.num_turns=$num_turns \
    actor_rollout_ref.rollout.rollout_type=$rollout_type \
    actor_rollout_ref.rollout.val_kwargs.n=8 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.top_k=-1 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.95 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.6 \
    actor_rollout_ref.rollout.val_kwargs.num_turns=2 \
    reward_model.policy_rs=$policy_rs \
    reward_model.rs_coef=$rs_coef \
    reward_model.split_verify_reward=$split_verify_reward \
    reward_model.feedback_mode=$feedback_mode \
    reward_model.lambda_regen=$lambda_regen \
    reward_model.lambda_cw=$lambda_cw \
    curriculum.enabled=$curriculum_enabled \
    curriculum.epsilon=$curriculum_epsilon \
    actor_rollout_ref.rollout.generic_counterfactual=$generic_counterfactual \
    actor_rollout_ref.rollout.include_generic_in_actor=False \
    critic.optim.lr=2e-6 \
    critic.use_dynamic_bsz=True \
    critic.model.use_remove_padding=True \
    "critic.model.path='${MODEL_PATH}'" \
    critic.model.fsdp_config.param_offload=True \
    critic.model.fsdp_config.optimizer_offload=True \
    algorithm.use_kl_in_reward=False \
    algorithm.norm_type=$norm_type \
    trainer.logger=$LOGGER \
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.n_gpus_per_node=$N_GPUS \
    trainer.nnodes=$NNODES \
    trainer.save_freq=10 \
    trainer.test_freq=10 \
    trainer.total_epochs=40 \
    trainer.default_local_dir=$CKPT_PATH/$PROJECT_NAME/$EXPERIMENT_NAME \
    trainer.val_before_train=True \
    trainer.resume_mode=auto \
    trainer.log_val_generations=2 \
    "$@"
