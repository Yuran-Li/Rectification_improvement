#!/bin/bash
#SBATCH --job-name=c1-g-contrast
#SBATCH --account=def-bboulet_gpu
#SBATCH --partition=gpubase_bynode_b4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=h100:4
#SBATCH --cpus-per-task=48
#SBATCH --mem=0
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --exclude=fc10218

# Frozen-cohort g contrast for C1@dynamic-step-100 (same protocol as C4):
# turn-1 g, K=8, T=1.0, top_k=10000. Question is whether undersampled C1
# stays at g ~ 0.98–1.0 vs uniform, not whether Δg is near zero.
#
#   cd $SCRATCH/verify_then_rectify/Rectification_dynamic
#   mkdir -p logs
#   sbatch quick_start/eval_c1_g_contrast.slurm
#
# Do NOT sbatch --export=ALL.

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
if [[ "$(basename "$REPO_ROOT")" == "quick_start" ]]; then
  REPO_ROOT="$(cd "$REPO_ROOT/.." && pwd)"
fi
cd "$REPO_ROOT"
mkdir -p "$REPO_ROOT/logs" "$REPO_ROOT/results/c1_g"

module load StdEnv/2023 gcc/12.3 cuda/12.2 python/3.10.13
# shellcheck source=/dev/null
source /scratch/yuranli/virtualenvs/PAG/bin/activate_pag.sh

export HF_HOME="${HF_HOME:-/scratch/yuranli/hf-cache}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/hub}"
export VLLM_USE_V1=0
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export TOKENIZERS_PARALLELISM=true
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-48}"

HF_BASE="${HF_BASE:-$HF_HOME/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306}"
CKPT_ROOT="${CKPT_ROOT:-$REPO_ROOT/checkpoints/PAG-SEC}"
COHORT="${COHORT:-$REPO_ROOT/tools/c1_cohort_step100.json}"
OUT_DIR="${OUT_DIR:-$REPO_ROOT/results/c1_g}"

merge_one() {
  local exp="$1" step="$2"
  local local_dir="$CKPT_ROOT/${exp}/global_step_${step}/actor"
  local target="$CKPT_ROOT/${exp}/global_step_${step}/actor_hf"
  if [[ -f "$target/config.json" && -f "$target/tokenizer.json" && -f "$target/model.safetensors" ]]; then
    echo "[merge] skip existing $target"
    return 0
  fi
  echo "[merge] $exp step=$step -> $target"
  test -f "$local_dir/model_world_size_8_rank_0.pt"
  python "$REPO_ROOT/scripts/model_merger.py" \
    --backend fsdp \
    --hf_model_path "$HF_BASE" \
    --local_dir "$local_dir" \
    --target_dir "$target"
  python - << PY
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("$HF_BASE", trust_remote_code=True)
tok.save_pretrained("$target")
print("[merge] tokenizer copied to $target")
PY
}

eval_one() {
  local tag="$1" model="$2"
  local out="$OUT_DIR/${tag}.json"
  if [[ -f "$out" ]]; then
    echo "[eval] skip existing $out"
    return 0
  fi
  echo "[eval] $tag  $model"
  python "$REPO_ROOT/tools/eval_c4_g_contrast.py" \
    --model "$model" \
    --tokenizer "$model" \
    --cohort "$COHORT" \
    --tag "$tag" \
    --output "$out" \
    --K "${K:-8}" \
    --temperature 1.0 \
    --top_k 10000 \
    --top_p 1.0 \
    --max_tokens 2028 \
    --gpu_util 0.70
}

echo "[c1-g] host=$(hostname)"
nvidia-smi -L || true
echo "[c1-g] cohort=$COHORT"

merge_one qwen1p5b_sec_dynamic_fir 100
merge_one qwen1p5b_sec_dynamic_fir 400
merge_one qwen1p5b_uniform_disc 100
merge_one qwen1p5b_uniform_disc 400

fail=0
CUDA_VISIBLE_DEVICES=0 eval_one base "$HF_BASE" &
CUDA_VISIBLE_DEVICES=1 eval_one dynamic_100 "$CKPT_ROOT/qwen1p5b_sec_dynamic_fir/global_step_100/actor_hf" &
CUDA_VISIBLE_DEVICES=2 eval_one dynamic_400 "$CKPT_ROOT/qwen1p5b_sec_dynamic_fir/global_step_400/actor_hf" &
CUDA_VISIBLE_DEVICES=3 eval_one uniform_100 "$CKPT_ROOT/qwen1p5b_uniform_disc/global_step_100/actor_hf" &
wait || fail=1
CUDA_VISIBLE_DEVICES=0 eval_one uniform_400 "$CKPT_ROOT/qwen1p5b_uniform_disc/global_step_400/actor_hf"
wait || fail=1
if [[ "$fail" -ne 0 ]]; then
  echo "[c1-g] one or more evals failed"
  exit 1
fi

python "$REPO_ROOT/tools/eval_c4_g_contrast.py" --summarize "$OUT_DIR" --cohort "$COHORT"
echo "[c1-g] done"
cat "$OUT_DIR/summary.json"
