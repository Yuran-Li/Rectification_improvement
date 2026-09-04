#!/usr/bin/env bash
# Run on the LAB SERVER (tmux). Pulls eval code + merged actor_hf from Fir.
# Does NOT copy FSDP shards / critic / wandb.
#
#   tmux new -s firpull
#   DEST=yuranli@fir.alliancecan.ca \
#   REMOTE=/scratch/yuranli/verify_then_rectify/Rectification_dynamic \
#   LOCAL=/path/on/lab/Rectification_dynamic \
#   bash sync_evals_to_lab.sh
#
# Optional: also pull uni@200/300 FSDP actor (19G each) if actor_hf is missing:
#   ALSO_SHARDS=1 bash sync_evals_to_lab.sh

set -euo pipefail

DEST="${DEST:-yuranli@fir.alliancecan.ca}"
REMOTE="${REMOTE:-/scratch/yuranli/verify_then_rectify/Rectification_dynamic}"
LOCAL="${LOCAL:-$PWD}"
HF_REMOTE="${HF_REMOTE:-/scratch/yuranli/hf-cache/hub/models--Qwen--Qwen2.5-1.5B-Instruct}"
HF_LOCAL="${HF_LOCAL:-$HOME/hf-cache/hub/models--Qwen--Qwen2.5-1.5B-Instruct}"

mkdir -p "$LOCAL" "$HF_LOCAL"

echo "[sync] code+scripts+cohorts+parquet  $DEST:$REMOTE -> $LOCAL"
rsync -avP --partial \
  --exclude 'checkpoints/' \
  --exclude 'wandb/' \
  --exclude 'logs/' \
  --exclude 'results/' \
  --exclude '__pycache__/' \
  --exclude '*.pyc' \
  --exclude '.git/' \
  "$DEST:$REMOTE/" "$LOCAL/"

pull_hf() {
  local exp="$1" step="$2"
  local rel="checkpoints/PAG-SEC/${exp}/global_step_${step}/actor_hf"
  echo "[sync] $rel"
  mkdir -p "$LOCAL/$rel"
  rsync -avP --partial "$DEST:$REMOTE/$rel/" "$LOCAL/$rel/"
}

pull_hf qwen1p5b_sec_dynamic_fir 100
pull_hf qwen1p5b_sec_dynamic_fir 400
pull_hf qwen1p5b_uniform_disc 100
pull_hf qwen1p5b_uniform_disc 400

if [[ "${ALSO_SHARDS:-0}" == "1" ]]; then
  for step in 200 300; do
    rel="checkpoints/PAG-SEC/qwen1p5b_uniform_disc/global_step_${step}/actor"
    echo "[sync] FSDP $rel (merge on lab afterwards)"
    mkdir -p "$LOCAL/$rel"
    rsync -avP --partial "$DEST:$REMOTE/$rel/" "$LOCAL/$rel/"
  done
else
  # If Fir already merged these, pull the small hf dirs.
  for step in 200 300; do
    rel="checkpoints/PAG-SEC/qwen1p5b_uniform_disc/global_step_${step}/actor_hf"
    if ssh "$DEST" "test -f $REMOTE/$rel/model.safetensors"; then
      echo "[sync] $rel"
      mkdir -p "$LOCAL/$rel"
      rsync -avP --partial "$DEST:$REMOTE/$rel/" "$LOCAL/$rel/"
    else
      echo "[sync] skip uni@${step} actor_hf (not merged). Re-run with ALSO_SHARDS=1 or merge on Fir."
    fi
  done
fi

echo "[sync] HF 1.5B Instruct"
mkdir -p "$(dirname "$HF_LOCAL")"
rsync -avP --partial "$DEST:$HF_REMOTE/" "$HF_LOCAL/"

echo "[sync] done."
echo "  cd $LOCAL"
echo "  bash quick_start/run_eval_lab.sh c1"
echo "  bash quick_start/run_eval_lab.sh c235"
echo "  bash quick_start/run_eval_lab.sh ladder   # needs uni@200/300 actor_hf"
