#!/bin/bash
#SBATCH --job-name=uni-g-ladder
#SBATCH --account=def-bboulet_gpu
#SBATCH --partition=gpubase_bynode_b4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=h100:4
#SBATCH --cpus-per-task=48
#SBATCH --mem=0
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --exclude=fc10218

# Uniform hindsight g ladder on ALL train prompts (n=7493).
# K=4, T=1.0, top_k=10000. Tags: base / uniform_{100,200,300,400}.
#
#   cd $SCRATCH/verify_then_rectify/Rectification_dynamic
#   mkdir -p logs
#   sbatch quick_start/eval_uniform_g_ladder.slurm
#
# Do NOT sbatch --export=ALL.

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
if [[ "$(basename "$REPO_ROOT")" == "quick_start" ]]; then
  REPO_ROOT="$(cd "$REPO_ROOT/.." && pwd)"
fi
cd "$REPO_ROOT"
mkdir -p "$REPO_ROOT/logs" "$REPO_ROOT/results/uniform_g_ladder"

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
EXP="${EXP:-qwen1p5b_uniform_disc}"
COHORT="${COHORT:-$REPO_ROOT/tools/train_all_cohort.json}"
OUT_DIR="${OUT_DIR:-$REPO_ROOT/results/uniform_g_ladder}"

merge_one() {
  local step="$1"
  local local_dir="$CKPT_ROOT/${EXP}/global_step_${step}/actor"
  local target="$CKPT_ROOT/${EXP}/global_step_${step}/actor_hf"
  if [[ -f "$target/config.json" && -f "$target/tokenizer.json" && -f "$target/model.safetensors" ]]; then
    echo "[merge] skip existing $target"
    return 0
  fi
  echo "[merge] $EXP step=$step -> $target"
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
    --K "${K:-4}" \
    --temperature 1.0 \
    --top_k 10000 \
    --top_p 1.0 \
    --max_tokens 2028 \
    --gpu_util 0.70
}

echo "[uni-g] host=$(hostname)"
nvidia-smi -L || true
echo "[uni-g] cohort=$COHORT"

test -f "$COHORT"

merge_one 100
merge_one 200
merge_one 300
merge_one 400

fail=0
CUDA_VISIBLE_DEVICES=0 eval_one base "$HF_BASE" &
CUDA_VISIBLE_DEVICES=1 eval_one uniform_100 "$CKPT_ROOT/${EXP}/global_step_100/actor_hf" &
CUDA_VISIBLE_DEVICES=2 eval_one uniform_200 "$CKPT_ROOT/${EXP}/global_step_200/actor_hf" &
CUDA_VISIBLE_DEVICES=3 eval_one uniform_300 "$CKPT_ROOT/${EXP}/global_step_300/actor_hf" &
wait || fail=1
CUDA_VISIBLE_DEVICES=0 eval_one uniform_400 "$CKPT_ROOT/${EXP}/global_step_400/actor_hf"
wait || fail=1
if [[ "$fail" -ne 0 ]]; then
  echo "[uni-g] one or more evals failed"
  exit 1
fi

python "$REPO_ROOT/tools/build_hindsight_stages.py" --eval-dir "$OUT_DIR"
echo "[uni-g] done"
python - << PY
import json
from pathlib import Path
p = Path("$OUT_DIR") / "hindsight_stages.json"
d = json.loads(p.read_text())
print("n", d["n"])
for k, v in d["stages"].items():
    print(k, v["sizes"])
PY
