#!/bin/bash
#SBATCH --job-name=sec-dyn-qwen1p5b
#SBATCH --account=def-bboulet_gpu
#SBATCH --partition=gpubase_bynode_b4
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=h100:4
#SBATCH --cpus-per-task=48
#SBATCH --mem=0
#SBATCH --time=3-00:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --exclude=fc10218

# Fir comparison vs Rectification_improvement (fixed MATH L1–L5 SEC).
# This worktree: dynamic C1–C5 SEC. Everything else matches the live
# fixed-SEC job 57305083 so the only intended difference is category
# definition + refresh.
#
# Submit from THIS repo root:
#   cd $SCRATCH/verify_then_rectify/Rectification_dynamic
#   mkdir -p logs
#   sbatch quick_start/train_qwen1p5b_sec_fir.slurm
#
# Uniform baseline (no SEC, no curriculum) — MUST rename the experiment
# or it writes into the dynamic-SEC checkpoint / wandb run:
#   sbatch --export=ALL,SEC_ENABLED=false,CURRICULUM_ENABLED=false,EXPERIMENT_NAME=qwen1p5b_uniform_disc,FEEDBACK_MODE=disc \
#     quick_start/train_qwen1p5b_sec_fir.slurm
#
# n=8 dynamic SEC (rollout.n and refresh K stay equal). New experiment name
# required so it does not resume the n=4 run:
#   sbatch --export=ALL,N_ROLLOUTS=8,SEC_REFRESH_ROLLOUTS=8,EXPERIMENT_NAME=qwen1p5b_sec_dynamic_n8,FEEDBACK_MODE=disc \
#     quick_start/train_qwen1p5b_sec_fir.slurm
#
# Smoke (1 train step):
#   sbatch --time=1:00:00 --partition=gpubase_bynode_b1 \
#     --export=ALL,SMOKE=1 \
#     quick_start/train_qwen1p5b_sec_fir.slurm

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
if [[ "$(basename "$REPO_ROOT")" == "quick_start" ]]; then
  REPO_ROOT="$(cd "$REPO_ROOT/.." && pwd)"
fi
cd "$REPO_ROOT"
mkdir -p "$REPO_ROOT/logs" "$REPO_ROOT/checkpoints"

module load StdEnv/2023 gcc/12.3 cuda/12.2 python/3.10.13
# shellcheck source=/dev/null
source /scratch/yuranli/virtualenvs/PAG/bin/activate_pag.sh
# ray_bootstrap_narval.sh lives in PAG-reproduction; reuse it.
# shellcheck source=/dev/null
source /scratch/yuranli/verify_then_rectify/PAG-reproduction/quick_start/ray_bootstrap_narval.sh

# ── HF / paths ────────────────────────────────────────────────────────────────
export HF_HOME="${HF_HOME:-/scratch/yuranli/hf-cache}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/hub}"
export CKPT_PATH="${CKPT_PATH:-$REPO_ROOT/checkpoints}"

# ── model / world size ────────────────────────────────────────────────────────
export MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-1.5B-Instruct}"
# Fir: 2 nodes × 4 H100 = 8 GPUs total.
# qwen1p5b_pag.sh hardcodes nnodes=1 n_gpus_per_node=8;
# we pass corrected values as extra hydra args below.
export N_GPUS=4
export NNODES=2

# ── experiment (distinct from qwen1p5b_sec_fixed) ─────────────────────────────
export PROJECT_NAME="${PROJECT_NAME:-PAG-SEC}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-qwen1p5b_sec_dynamic_fir}"
export USE_WANDB="${USE_WANDB:-1}"
export SMOKE="${SMOKE:-0}"

# ── SEC (dynamic C1–C5). α/τ match fixed-SEC. ─────────────────────────────────
# Overridable so the same script can launch the uniform baseline:
#   --export=ALL,SEC_ENABLED=false,EXPERIMENT_NAME=qwen1p5b_uniform_disc
export SEC_ENABLED="${SEC_ENABLED:-true}"
export CURRICULUM_ENABLED="${CURRICULUM_ENABLED:-false}"
export SEC_Q_ALPHA="${SEC_Q_ALPHA:-0.1}"
export SEC_TEMPERATURE="${SEC_TEMPERATURE:-1.0}"
export SEC_REFRESH_INTERVAL="${SEC_REFRESH_INTERVAL:-50}"
# rollout.n and SEC refresh K must match (trainer asserts this).
export N_ROLLOUTS="${N_ROLLOUTS:-8}"
export SEC_REFRESH_ROLLOUTS="${SEC_REFRESH_ROLLOUTS:-$N_ROLLOUTS}"
[[ -n "${SEC_INITIAL_CATEGORY_STATS:-}" ]] && export SEC_INITIAL_CATEGORY_STATS

# Comparison contract with Rectification_improvement (must match except
# dynamic C1–C5 + refresh). Default disc matches job 57305083.
export FEEDBACK_MODE="${FEEDBACK_MODE:-disc}"

# ── optional leftover overrides ───────────────────────────────────────────────
[[ -n "${LAMBDA_REGEN:-}"           ]] && export LAMBDA_REGEN
[[ -n "${GENERIC_COUNTERFACTUAL:-}" ]] && export GENERIC_COUNTERFACTUAL

# ── Ray setup ─────────────────────────────────────────────────────────────────
nodes=($(scontrol show hostnames "$SLURM_JOB_NODELIST"))
head_node="${nodes[0]}"
port=6379
echo "[sec-dyn-fir] resolving head IP via srun..."
head_ip="$(ray_resolve_head_ip "$head_node")"
echo "[sec-dyn-fir] head=${head_node} ip=${head_ip} nodes=${nodes[*]}"
echo "[sec-dyn-fir] MODEL_PATH=${MODEL_PATH}  N_GPUS=${N_GPUS}  NNODES=${NNODES}"
echo "[sec-dyn-fir] SEC_ENABLED=${SEC_ENABLED}  SEC_Q_ALPHA=${SEC_Q_ALPHA}  SEC_TEMPERATURE=${SEC_TEMPERATURE}"
echo "[sec-dyn-fir] N_ROLLOUTS=${N_ROLLOUTS}  SEC_REFRESH_INTERVAL=${SEC_REFRESH_INTERVAL}  SEC_REFRESH_ROLLOUTS=${SEC_REFRESH_ROLLOUTS}"
echo "[sec-dyn-fir] FEEDBACK_MODE=${FEEDBACK_MODE}  CURRICULUM_ENABLED=${CURRICULUM_ENABLED}"
echo "[sec-dyn-fir] EXPERIMENT_NAME=${EXPERIMENT_NAME}  PROJECT_NAME=${PROJECT_NAME}  USE_WANDB=${USE_WANDB}"
echo "[sec-dyn-fir] CKPT_PATH=${CKPT_PATH}/${PROJECT_NAME}/${EXPERIMENT_NAME}"

# ── GPU health check ──────────────────────────────────────────────────────────
echo "[sec-dyn-fir] Checking GPU health on all nodes..."
srun --nodes="$SLURM_NNODES" --ntasks="$SLURM_NNODES" --ntasks-per-node=1 \
  bash -c '
    failed=0
    while IFS="," read -r idx free total; do
      free=$(echo $free | tr -d " "); total=$(echo $total | tr -d " ")
      echo "[gpu-check] ${SLURMD_NODENAME} GPU${idx}: ${free} MiB free / ${total} MiB total"
      if (( free < 70000 )); then
        echo "[gpu-check] ERROR: ${SLURMD_NODENAME} GPU${idx} has only ${free} MiB free." \
             "Possible stale/zombie process! Use nvidia-smi on that node to investigate." >&2
        failed=1
      fi
    done < <(nvidia-smi --query-gpu=index,memory.free,memory.total --format=csv,noheader,nounits)
    exit $failed
  '
echo "[sec-dyn-fir] GPU health check passed on all nodes -- proceeding."

# ── Start Ray on all nodes ────────────────────────────────────────────────────
srun --nodes="$SLURM_NNODES" --ntasks="$SLURM_NNODES" --ntasks-per-node=1 \
  bash -lc "
    set -euo pipefail
    source /scratch/yuranli/virtualenvs/PAG/bin/activate_pag.sh
    source '/scratch/yuranli/verify_then_rectify/PAG-reproduction/quick_start/ray_bootstrap_narval.sh'
    if [[ \"\${SLURMD_NODENAME}\" == '${head_node}' ]]; then
      ray_start_head '${head_ip}' '${port}' '${N_GPUS}'
    else
      sleep 10
      ray_start_worker '${head_ip}' '${port}' '${N_GPUS}'
    fi
    sleep infinity
  " &
SRUN_RAY_PID=$!

export RAY_ADDRESS="${head_ip}:${port}"
sleep 5
ray_wait_ready "$RAY_ADDRESS" 60

cleanup() {
  echo '[sec-dyn-fir] shutting down ray...'
  kill "$SRUN_RAY_PID" 2>/dev/null || true
  srun --nodes="$SLURM_NNODES" --ntasks="$SLURM_NNODES" --ntasks-per-node=1 \
    bash -lc 'source /scratch/yuranli/virtualenvs/PAG/bin/activate_pag.sh; ray stop --force || true' \
    || true
}
trap cleanup EXIT

echo "[sec-dyn-fir] RAY_ADDRESS=${RAY_ADDRESS}"
ray status --address="$RAY_ADDRESS" || true

# ── Run training ──────────────────────────────────────────────────────────────
# Run directly in batch-script context (same node as Ray head).
# Using srun here would exhaust the task slot held by the background ray srun.
export RAY_ADDRESS RAY_GCS_SERVER_REQUEST_TIMEOUT_SECONDS=120
export N_GPUS NNODES MODEL_PATH CKPT_PATH
export HF_HOME HF_HUB_CACHE TRANSFORMERS_CACHE
export USE_WANDB PROJECT_NAME EXPERIMENT_NAME
export SEC_ENABLED SEC_Q_ALPHA SEC_TEMPERATURE
export SEC_REFRESH_INTERVAL SEC_REFRESH_ROLLOUTS N_ROLLOUTS
export FEEDBACK_MODE CURRICULUM_ENABLED
export VLLM_USE_V1=0
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export TOKENIZERS_PARALLELISM=true
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export PYTHONNOUSERSITE=1

# Extra hydra overrides match Rectification_improvement/train_qwen1p5b_sec_fir.slurm:
#   - world size 2×4
#   - ppo_max_token_len_per_gpu=16384 (must match A6000 / fixed-SEC)
#   - max_num_batched_tokens=16384
#   - gpu_memory_utilization=0.3
#   - actor/ref/critic offload + use_torch_compile=False (Fir H100 OOM engineering)
EXTRA_ARGS=(
  trainer.n_gpus_per_node=${N_GPUS}
  trainer.nnodes=${NNODES}
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=16384
  actor_rollout_ref.rollout.max_num_batched_tokens=16384
  actor_rollout_ref.rollout.gpu_memory_utilization=${GPU_MEM_UTIL:-0.3}
  actor_rollout_ref.actor.use_torch_compile=False
  actor_rollout_ref.actor.fsdp_config.param_offload=True
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=True
  actor_rollout_ref.ref.fsdp_config.param_offload=True
  critic.model.fsdp_config.param_offload=True
  critic.model.fsdp_config.optimizer_offload=True
)
if [[ "${SMOKE:-0}" == "1" ]]; then
  EXTRA_ARGS+=(trainer.total_training_steps=1)
fi

set +e
bash "$REPO_ROOT/quick_start/qwen1p5b_pag.sh" "${EXTRA_ARGS[@]}"
status=$?
set -e

if [[ $status -eq 0 ]]; then
  echo "[sec-dyn-fir] PASS (exit=0)."
else
  echo "[sec-dyn-fir] FAIL (exit=${status}). Check logs for CUDA OOM / Ray / vLLM / NCCL errors."
fi
exit "$status"
