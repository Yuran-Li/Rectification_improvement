#!/bin/bash
#SBATCH --job-name=pag-qwen1p5b
#SBATCH --account=def-bboulet_gpu
#SBATCH --partition=gpubase_bynode_b4
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=0
#SBATCH --time=2-12:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# Narval: 2 nodes × 4×A100 = 8 GPUs (closest to upstream single-node 8-GPU).
# Submit from repo root or quick_start:
#   mkdir -p logs
#   sbatch quick_start/train_pag_narval.slurm
#
# Optional overrides at submit time:
#   sbatch --export=ALL,USE_WANDB=1,EXPERIMENT_NAME=my_run quick_start/train_pag_narval.slurm

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
# If submitted from quick_start/, prefer parent
if [[ "$(basename "$REPO_ROOT")" == "quick_start" ]]; then
  REPO_ROOT="$(cd "$REPO_ROOT/.." && pwd)"
fi
cd "$REPO_ROOT"
mkdir -p "$REPO_ROOT/logs" "$REPO_ROOT/checkpoints"

module load StdEnv/2023 gcc/12.3 cuda/12.2 python/3.10.13
source /scratch/yuranli/virtualenvs/PAG/bin/activate_pag.sh

export HF_HOME="${HF_HOME:-/scratch/yuranli/hf-cache}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/hub}"
export CKPT_PATH="${CKPT_PATH:-$REPO_ROOT/checkpoints}"
# Hub id or local snapshot; download once to HF_HOME before long runs if possible
export MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-1.5B-Instruct}"
export N_GPUS=4
export NNODES=2
export USE_WANDB="${USE_WANDB:-0}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-qwen1p5b_pag_narval8}"

# Shared Ray temp under job-local scratch when available
export RAY_TMPDIR="${RAY_TMPDIR:-${SLURM_TMPDIR:-/tmp}/ray}"
mkdir -p "$RAY_TMPDIR"

nodes=($(scontrol show hostnames "$SLURM_JOB_NODELIST"))
head_node="${nodes[0]}"
port=6379
head_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname -i | awk '{print $1}')
echo "[slurm] head=${head_node} ip=${head_ip} nodes=${nodes[*]}"

# Start Ray on every allocated node (head + workers)
srun --nodes="$SLURM_NNODES" --ntasks="$SLURM_NNODES" --ntasks-per-node=1 \
  bash -c "
    set -euo pipefail
    source /scratch/yuranli/virtualenvs/PAG/bin/activate_pag.sh
    export PYTHONNOUSERSITE=1
    export VLLM_USE_V1=0
    export VLLM_WORKER_MULTIPROC_METHOD=spawn
    export TOKENIZERS_PARALLELISM=true
    export RAY_TMPDIR='${RAY_TMPDIR}'
    if [[ \"\${SLURMD_NODENAME}\" == '${head_node}' ]]; then
      ray start --head --node-ip-address='${head_ip}' --port=${port} \
        --num-cpus=\${SLURM_CPUS_PER_TASK} --num-gpus=${N_GPUS} --block
    else
      sleep 20
      ray start --address='${head_ip}:${port}' \
        --num-cpus=\${SLURM_CPUS_PER_TASK} --num-gpus=${N_GPUS} --block
    fi
  " &
RAY_SRUN_PID=$!

cleanup() {
  echo '[slurm] shutting down ray...'
  srun --nodes="$SLURM_NNODES" --ntasks="$SLURM_NNODES" --ntasks-per-node=1 \
    bash -c 'source /scratch/yuranli/virtualenvs/PAG/bin/activate_pag.sh; ray stop --force || true' \
    || true
  kill "$RAY_SRUN_PID" 2>/dev/null || true
}
trap cleanup EXIT

# Give the cluster a moment to come up
sleep 45
export RAY_ADDRESS="${head_ip}:${port}"
echo "[slurm] RAY_ADDRESS=${RAY_ADDRESS}"
ray status --address="$RAY_ADDRESS" || true

# Launch trainer on the head node only
srun --nodes=1 --ntasks=1 -w "$head_node" \
  env RAY_ADDRESS="$RAY_ADDRESS" \
      N_GPUS="$N_GPUS" \
      NNODES="$NNODES" \
      MODEL_PATH="$MODEL_PATH" \
      CKPT_PATH="$CKPT_PATH" \
      HF_HOME="$HF_HOME" \
      HF_HUB_CACHE="$HF_HUB_CACHE" \
      TRANSFORMERS_CACHE="$TRANSFORMERS_CACHE" \
      USE_WANDB="$USE_WANDB" \
      EXPERIMENT_NAME="$EXPERIMENT_NAME" \
      bash "$REPO_ROOT/quick_start/run_pag_local.sh"

echo '[slurm] training finished'
