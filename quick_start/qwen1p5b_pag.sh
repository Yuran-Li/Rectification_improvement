set -x

math500=datasets/math500.parquet
math7500=datasets/math7500.parquet
aime2024=datasets/aime2024.parquet
aime2025=datasets/aime2025.parquet
minervamath=datasets/minervamath.parquet
dapo17k=datasets/dapo17k.parquet

PROJECT_NAME="${PROJECT_NAME:-PAG}"
CKPT_PATH="${CKPT_PATH:-checkpoints}"
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-1.5B-Instruct}"

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
echo "[qwen1p5b_pag] MODEL_PATH=$MODEL_PATH  training_dataset=$training_dataset"

EXPERIMENT_NAME="${EXPERIMENT_NAME:-qwen1p5b_pag}"
# Train-time group size. Override with N_ROLLOUTS (or SEC_REFRESH_ROLLOUTS).
# SEC refresh K must equal this; default both to 8.
n="${N_ROLLOUTS:-${SEC_REFRESH_ROLLOUTS:-8}}"
rollout_type=pag
num_turns=2
policy_rs=True
rs_coef=1.0
norm_type=role
split_verify_reward=True
# generic | regen | delta | acc | disc
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
# CURRICULUM_ENABLED=true  → GenerationFrontierSampler
# CURRICULUM_ENABLED=false → uniform RandomSampler (baseline)
curriculum_enabled="${CURRICULUM_ENABLED:-false}"
curriculum_epsilon="${CURRICULUM_EPSILON:-0.3}"

# SEC (Self-Evolving Curriculum) — mutually exclusive with CURRICULUM_ENABLED.
# SEC_ENABLED=true  → dynamic C1–C5 + online Q update
# SEC_ENABLED=false → use CURRICULUM_ENABLED / uniform baseline
sec_enabled="${SEC_ENABLED:-false}"
sec_q_alpha="${SEC_Q_ALPHA:-0.1}"
sec_temperature="${SEC_TEMPERATURE:-1.0}"
sec_refresh_interval="${SEC_REFRESH_INTERVAL:-50}"
sec_refresh_rollouts="${SEC_REFRESH_ROLLOUTS:-$n}"
sec_initial_stats="${SEC_INITIAL_CATEGORY_STATS:-null}"

if [[ "$sec_enabled" == "true" && "$curriculum_enabled" == "true" ]]; then
  echo "ERROR: SEC_ENABLED and CURRICULUM_ENABLED cannot both be true." >&2
  exit 1
fi
if [[ "$sec_enabled" == "true" && "$sec_refresh_rollouts" != "$n" ]]; then
  echo "ERROR: SEC_REFRESH_ROLLOUTS (${sec_refresh_rollouts}) must equal N_ROLLOUTS/n (${n})." >&2
  exit 1
fi

if [[ -n "${GENERIC_COUNTERFACTUAL:-}" ]]; then
  generic_counterfactual="$GENERIC_COUNTERFACTUAL"
elif [[ "$feedback_mode" == "generic" ]]; then
  generic_counterfactual=True
else
  generic_counterfactual=False
fi

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
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
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
    sec.enabled=$sec_enabled \
    sec.q_alpha=$sec_q_alpha \
    sec.temperature=$sec_temperature \
    sec.refresh_interval=$sec_refresh_interval \
    sec.refresh_rollouts=$sec_refresh_rollouts \
    sec.initial_category_stats_path=$sec_initial_stats \
    actor_rollout_ref.rollout.generic_counterfactual=$generic_counterfactual \
    actor_rollout_ref.rollout.include_generic_in_actor=False \
    critic.optim.lr=2e-6 \
    critic.use_dynamic_bsz=True \
    critic.model.use_remove_padding=True \
    "critic.model.path='${MODEL_PATH}'" \
    critic.model.fsdp_config.param_offload=False \
    critic.model.fsdp_config.optimizer_offload=False \
    algorithm.use_kl_in_reward=False \
    algorithm.norm_type=$norm_type \
    trainer.logger=['console','wandb'] \
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=10 \
    trainer.test_freq=10 \
    trainer.total_epochs=40 \
    trainer.default_local_dir=$CKPT_PATH/$PROJECT_NAME/$EXPERIMENT_NAME \
    trainer.val_before_train=True \
    trainer.resume_mode=auto \
    trainer.log_val_generations=2 \
    "$@"
