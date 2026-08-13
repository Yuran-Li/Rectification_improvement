#!/usr/bin/env bash
# Feasibility-guided PAG RL (role-aware state-level V_F).
#
# Formal objective: F(s)=V_F(s)-ε gates recovery supervision (default: no global CMDP).
#   s^V=(x,y_i), s^R=(x,y_i,v_i); G_F=1[eventual fail in remaining horizon].
#   Self-bootstrap = problem-conditioned replay; GPT = state-conditioned a_E|s_A.
#
# Example:
#   N_GPUS=8 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
#     bash quick_start/run_feasibility_pag.sh
#
# Smoke (1 GPU, tiny batch):
#   N_GPUS=1 TRAIN_BS=8 MINI_BS=8 TOTAL_EPOCHS=1 \
#     bash quick_start/run_feasibility_pag.sh
set -euo pipefail
set -x

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

export HF_HOME="${HF_HOME:-${SCRATCH:-/scratch/$USER}/hf-cache}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/hub}"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_USE_V1=0
export TOKENIZERS_PARALLELISM=true
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export PYTHONNOUSERSITE=1

math500="$REPO_ROOT/datasets/math500.parquet"
math7500="$REPO_ROOT/datasets/math7500.parquet"

PROJECT_NAME="${PROJECT_NAME:-Rectification_Feasibility}"
CKPT_PATH="${CKPT_PATH:-$REPO_ROOT/checkpoints}"
# Optional SFT init; leave empty / unset to start from BASE_MODEL.
SFT_RECTIFY="${SFT_RECTIFY:-}"
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-7B-Instruct}"
if [[ -n "${SFT_RECTIFY}" && -d "${SFT_RECTIFY}" ]]; then
  MODEL_PATH="${MODEL_PATH:-$SFT_RECTIFY}"
else
  MODEL_PATH="${MODEL_PATH:-$BASE_MODEL}"
fi

EXPERIMENT_NAME="${EXPERIMENT_NAME:-qwen25_7b_instruct_feas_pag_t2}"
n="${N_SAMPLES:-4}"
rollout_type=pag
num_turns="${NUM_TURNS:-2}"
slide_window="${SLIDE_WINDOW:-False}"  # False: full-concat traj (default); True: Markov pack (legacy)
utility_aware="${UTILITY_AWARE:-True}"
alpha="${ALPHA:-1.0}"
beta="${BETA:-0.5}"
# Role-aware V_F + F(s)=V_F-ε gate (default: no global CMDP).
dual_critic="${DUAL_CRITIC:-True}"
use_lagrangian="${USE_LAGRANGIAN:-False}"
critic_num_labels="${CRITIC_NUM_LABELS:-2}"
# ε in F(s)=V_F(s)-ε. Prefer FEAS_THRESHOLD; COST_BUDGET kept as alias.
feas_threshold="${FEAS_THRESHOLD:-${COST_BUDGET:-0.3}}"
cost_budget="$feas_threshold"
# First N steps: no infeasible BC/GPT; V_F still trains
constraint_warmup="${CONSTRAINT_WARMUP:-15}"
# F(s)>ε → PPO + role recovery BC; F≤0 → PPO only (light BC default 0)
expert_bc="${EXPERT_BC:-True}"
expert_bc_coef="${EXPERT_BC_COEF:-2.0}"
expert_bc_light="${EXPERT_BC_LIGHT:-0.0}"
expert_buffer_capacity="${EXPERT_BUFFER_CAPACITY:-256}"
# Same-problem bootstrap: problem-conditioned positive replay (n>1). No API.
bootstrap_same_uid="${BOOTSTRAP_SAME_UID:-True}"
# Online GPT: state-conditioned a_E|s_A when F(s)>ε and no bootstrap expert
online_gpt_expert="${ONLINE_GPT_EXPERT:-False}"
online_gpt_model="${ONLINE_GPT_MODEL:-gpt-4o-mini}"
online_gpt_max_per_step="${ONLINE_GPT_MAX_PER_STEP:-4}"
online_gpt_prefer_bootstrap="${ONLINE_GPT_PREFER_BOOTSTRAP:-True}"
norm_type="${NORM_TYPE:-role}"

TRAIN_BS="${TRAIN_BS:-32}"
MINI_BS="${MINI_BS:-32}"
MAX_PROMPT="${MAX_PROMPT:-1024}"
MAX_RESP="${MAX_RESP:-1024}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-20}"
# Optional hard stop (overrides epochs). Example: TOTAL_TRAINING_STEPS=25
TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-}"
# Critic-only VF smoke: CRITIC_ONLY=1 → never update actor (only rollout + critic BCE)
CRITIC_WARMUP="${CRITIC_WARMUP:-0}"
if [[ "${CRITIC_ONLY:-0}" == "1" ]]; then
  CRITIC_WARMUP=999999
  EXPERT_BC="${EXPERT_BC:-False}"
  expert_bc=False
  echo "[run] CRITIC_ONLY=1 → trainer.critic_warmup=$CRITIC_WARMUP expert_bc=False"
fi
N_GPUS="${N_GPUS:-8}"
EXTRA_TRAINER_ARGS=()
if [[ -n "${TOTAL_TRAINING_STEPS}" ]]; then
  EXTRA_TRAINER_ARGS+=(trainer.total_training_steps="$TOTAL_TRAINING_STEPS")
  echo "[run] total_training_steps=$TOTAL_TRAINING_STEPS"
fi
NNODES="${NNODES:-1}"
# Align train sampling with val (fewer degenerate garbage gens)
TEMPERATURE="${TEMPERATURE:-0.6}"
TOP_P="${TOP_P:-0.95}"
TOP_K="${TOP_K:--1}"
# Hybrid engine: free vLLM KV/cudagraph before actor log_prob (needs enforce_eager).
# Previous default free_cache_engine=False + util=0.55 OOMs after val→train on 48G.
ENFORCE_EAGER="${ENFORCE_EAGER:-True}"
FREE_CACHE_ENGINE="${FREE_CACHE_ENGINE:-True}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.40}"
VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-True}"
PPO_MAX_TOKEN_LEN="${PPO_MAX_TOKEN_LEN:-12288}"

if [[ "${USE_WANDB:-0}" == "1" ]]; then
  LOGGER="['console','wandb']"
else
  LOGGER="['console']"
fi

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=gae \
    data.train_files=[$math7500] \
    data.val_files="['$math500']" \
    data.filter_overlong_prompts=True \
    data.train_batch_size=$TRAIN_BS \
    data.max_prompt_length=$MAX_PROMPT \
    data.max_response_length=$MAX_RESP \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$PPO_MAX_TOKEN_LEN \
    actor_rollout_ref.rollout.max_num_batched_tokens=$PPO_MAX_TOKEN_LEN \
    actor_rollout_ref.rollout.enforce_eager=$ENFORCE_EAGER \
    actor_rollout_ref.rollout.free_cache_engine=$FREE_CACHE_ENGINE \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0 \
    actor_rollout_ref.actor.ppo_mini_batch_size=$MINI_BS \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.clip_ratio_high=0.28 \
    actor_rollout_ref.actor.clip_ratio_low=0.2 \
    actor_rollout_ref.actor.expert_bc_coef=$expert_bc_coef \
    actor_rollout_ref.actor.expert_bc_light=$expert_bc_light \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=$GPU_MEM_UTIL \
    actor_rollout_ref.rollout.n=$n \
    actor_rollout_ref.rollout.temperature=$TEMPERATURE \
    actor_rollout_ref.rollout.top_p=$TOP_P \
    actor_rollout_ref.rollout.top_k=$TOP_K \
    actor_rollout_ref.rollout.num_turns=$num_turns \
    actor_rollout_ref.rollout.rollout_type=$rollout_type \
    actor_rollout_ref.rollout.slide_window=$slide_window \
    actor_rollout_ref.rollout.val_kwargs.n=4 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.top_k=$TOP_K \
    actor_rollout_ref.rollout.val_kwargs.top_p=$TOP_P \
    actor_rollout_ref.rollout.val_kwargs.temperature=$TEMPERATURE \
    actor_rollout_ref.rollout.val_kwargs.num_turns=$num_turns \
    reward_model.utility_aware=$utility_aware \
    reward_model.alpha=$alpha \
    reward_model.beta=$beta \
    critic.optim.lr=2e-6 \
    critic.use_dynamic_bsz=True \
    critic.model.use_remove_padding=True \
    critic.model.path=$MODEL_PATH \
    critic.model.num_labels=$critic_num_labels \
    critic.vf_loss_coef=1.0 \
    critic.model.fsdp_config.param_offload=True \
    critic.model.fsdp_config.optimizer_offload=True \
    algorithm.use_kl_in_reward=False \
    algorithm.norm_type=$norm_type \
    algorithm.dual_critic=$dual_critic \
    algorithm.use_lagrangian=$use_lagrangian \
    algorithm.cost_budget=$cost_budget \
    algorithm.constraint_warmup=$constraint_warmup \
    algorithm.expert_bc=$expert_bc \
    algorithm.expert_buffer_capacity=$expert_buffer_capacity \
    algorithm.bootstrap_same_uid=$bootstrap_same_uid \
    algorithm.online_gpt_expert=$online_gpt_expert \
    algorithm.online_gpt_model=$online_gpt_model \
    algorithm.online_gpt_max_per_step=$online_gpt_max_per_step \
    algorithm.online_gpt_prefer_bootstrap=$online_gpt_prefer_bootstrap \
    trainer.logger=$LOGGER \
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.n_gpus_per_node=$N_GPUS \
    trainer.nnodes=$NNODES \
    trainer.save_freq="${SAVE_FREQ:-20}" \
    trainer.test_freq="${TEST_FREQ:-10}" \
    trainer.total_epochs=$TOTAL_EPOCHS \
    trainer.default_local_dir=$CKPT_PATH/$PROJECT_NAME/$EXPERIMENT_NAME \
    trainer.val_before_train=$VAL_BEFORE_TRAIN \
    trainer.critic_warmup=$CRITIC_WARMUP \
    trainer.resume_mode="${RESUME_MODE:-auto}" \
    trainer.log_val_generations=2 \
    "${EXTRA_TRAINER_ARGS[@]}"
