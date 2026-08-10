#!/usr/bin/env bash
# Feasibility-guided PAG: slide-window multi-turn PPO (Phase 1).
#
# Defaults match the locked design:
#   - num_turns=4, slide_window=True (keep problem + latest answer only)
#   - utility_aware rewards (alpha/beta) + feasibility cost logging
#   - init from SFT rectify ckpt when present
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
SFT_RECTIFY="${SFT_RECTIFY:-$REPO_ROOT/checkpoints/sft/qwen25math7b_pag_sft_rectify/global_step_75}"
BASE_MODEL="${BASE_MODEL:-/data/yuranli/LLM/2026.04/models/Qwen2.5-Math-7B-Instruct}"
if [[ -d "$SFT_RECTIFY" ]]; then
  MODEL_PATH="${MODEL_PATH:-$SFT_RECTIFY}"
else
  MODEL_PATH="${MODEL_PATH:-$BASE_MODEL}"
fi

EXPERIMENT_NAME="${EXPERIMENT_NAME:-qwen25math7b_feas_pag_t4}"
n="${N_SAMPLES:-4}"
rollout_type=pag
num_turns="${NUM_TURNS:-3}"
slide_window="${SLIDE_WINDOW:-True}"
utility_aware="${UTILITY_AWARE:-True}"
alpha="${ALPHA:-1.0}"
beta="${BETA:-0.5}"
# Turn-level γ_F for correction rounds: Σ_t γ_F^{t-1} c_t^F  (pag.py c_f_discounted only).
# NOT the same as token-level GAE discount below.
gamma_f="${GAMMA_F:-0.9}"
# Phase-2 dual critic VR/VF + Lagrangian
dual_critic="${DUAL_CRITIC:-True}"
critic_num_labels="${CRITIC_NUM_LABELS:-2}"
lagrange_lambda_init="${LAGRANGE_LAMBDA_INIT:-0.0}"
lagrange_lr="${LAGRANGE_LR:-0.001}"
cost_budget="${COST_BUDGET:-0.3}"
# Token-level γ inside cost GAE (returns_f). Should be 1.0 like reward γ — sparse costs
# live at segment ends; using 0.9 here wrongly decays across response tokens.
cost_gamma="${COST_GAMMA:-1.0}"
# Pure PPO for first N steps (λ=0, no infeasible BC); V_F still trains
constraint_warmup="${CONSTRAINT_WARMUP:-50}"
# Phase-3: infeasible → PG + high-weight expert BC (do not turn off PG)
expert_bc="${EXPERT_BC:-True}"
expert_bc_coef="${EXPERT_BC_COEF:-2.0}"
# Light BC on successful expert spans even when feasible (helps no-API runs)
expert_bc_light="${EXPERT_BC_LIGHT:-0.2}"
expert_buffer_capacity="${EXPERT_BUFFER_CAPACITY:-256}"
# Same-problem bootstrap: copy success traj to infeasible siblings (n>1). No API.
bootstrap_same_uid="${BOOTSTRAP_SAME_UID:-True}"
# Online GPT: when V_F>B_F and no bootstrap expert, call GPT for that task then BC
# Requires OPENAI_API_KEY (optional OPENAI_BASE_URL). Default off for first ablations.
online_gpt_expert="${ONLINE_GPT_EXPERT:-False}"
online_gpt_model="${ONLINE_GPT_MODEL:-gpt-4o-mini}"
online_gpt_max_per_step="${ONLINE_GPT_MAX_PER_STEP:-4}"
online_gpt_prefer_bootstrap="${ONLINE_GPT_PREFER_BOOTSTRAP:-True}"
# legacy PAG shaping off when utility_aware; kept for ablations
policy_rs="${POLICY_RS:-False}"
rs_coef="${RS_COEF:-1.0}"
norm_type="${NORM_TYPE:-role}"

TRAIN_BS="${TRAIN_BS:-128}"
MINI_BS="${MINI_BS:-64}"
MAX_PROMPT="${MAX_PROMPT:-1024}"
MAX_RESP="${MAX_RESP:-1024}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-20}"
N_GPUS="${N_GPUS:-8}"
NNODES="${NNODES:-1}"
# Align train sampling with val (fewer degenerate garbage gens)
TEMPERATURE="${TEMPERATURE:-0.6}"
TOP_P="${TOP_P:-0.95}"
TOP_K="${TOP_K:--1}"
# Hybrid engine: free vLLM KV/cudagraph before actor log_prob (needs enforce_eager).
# Previous default free_cache_engine=False + util=0.55 OOMs after val→train on 48G.
ENFORCE_EAGER="${ENFORCE_EAGER:-True}"
FREE_CACHE_ENGINE="${FREE_CACHE_ENGINE:-True}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.45}"
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
    reward_model.policy_rs=$policy_rs \
    reward_model.rs_coef=$rs_coef \
    reward_model.utility_aware=$utility_aware \
    reward_model.alpha=$alpha \
    reward_model.beta=$beta \
    reward_model.gamma_f=$gamma_f \
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
    algorithm.lagrange_lambda_init=$lagrange_lambda_init \
    algorithm.lagrange_lr=$lagrange_lr \
    algorithm.cost_budget=$cost_budget \
    algorithm.cost_gamma=$cost_gamma \
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
    trainer.save_freq=50 \
    trainer.test_freq=10 \
    trainer.total_epochs=$TOTAL_EPOCHS \
    trainer.default_local_dir=$CKPT_PATH/$PROJECT_NAME/$EXPERIMENT_NAME \
    trainer.val_before_train=$VAL_BEFORE_TRAIN \
    trainer.resume_mode=disable \
    trainer.log_val_generations=2
