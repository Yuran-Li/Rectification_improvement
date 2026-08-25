#!/usr/bin/env bash
# End-to-end S2R-style SFT collection → PAG multi-turn parquet.
#
# Stages:
#   0) prepare_seed_jsonl.py      parquet/jsonl → seed
#   1) collect_solutions.py      local vLLM → y0
#   2) collect_verifications.py  GPT API → verify
#   2b) collect_y1.py            local vLLM + GPT critique → y1 (no extra GPT)
#   3) construct_s2r_sft.py      stitch; correct tail from gold y0 ∪ gold y1
#   4) convert_s2r_to_pag_multiturn.py → PAG parquet
#
# Prerequisites:
#   - vLLM server for step 1 (default localhost:8081)
#   - OPENAI_API_KEY for step 2
#   - PAG / project env with openai, transformers, pandas, pyarrow, tqdm
#
# Examples:
#   # smoke (200 MATH7500 prompts)
#   MAX_ROWS=200 bash tools/sft_data/run_collect_pipeline.sh
#
#   # skip GPT if verifications already exist
#   SKIP_VERIFY=1 bash tools/sft_data/run_collect_pipeline.sh
#
#   # only convert an existing S2R-style JSON
#   STAGES=convert S2R_JSON=datasets/sft_collect/Qwen-1.5B/sft_s2r_style.json \
#     PAG_OUT=datasets/sft/Qwen-1.5B bash tools/sft_data/run_collect_pipeline.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

sft_tag_from_model() {
  local m
  m="$(printf '%s' "$1" | tr '[:upper:]' '[:lower:]')"
  if [[ "$m" =~ (^|[^0-9])7b([^0-9]|$) ]]; then
    echo "Qwen-7B"
  else
    echo "Qwen-1.5B"
  fi
}

PY="${PY:-python}"
SFT_DIR="$REPO_ROOT/tools/sft_data"
SOLUTION_MODEL="${SOLUTION_MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"
SFT_TAG="${SFT_TAG:-$(sft_tag_from_model "$SOLUTION_MODEL")}"
OUT_DIR="${OUT_DIR:-$REPO_ROOT/datasets/sft_collect/$SFT_TAG}"
PAG_OUT="${PAG_OUT:-$REPO_ROOT/datasets/sft/$SFT_TAG}"
mkdir -p "$OUT_DIR" "$PAG_OUT"

SEED_PARQUET="${SEED_PARQUET:-$REPO_ROOT/datasets/math7500.parquet}"
SEED_FILE="${SEED_FILE:-$OUT_DIR/seed.jsonl}"
SOLUTIONS="${SOLUTIONS:-$OUT_DIR/solutions.jsonl}"
VERIFICATIONS="${VERIFICATIONS:-$OUT_DIR/verifications.jsonl}"
Y1="${Y1:-$OUT_DIR/y1.jsonl}"
S2R_JSON="${S2R_JSON:-$OUT_DIR/sft_s2r_style.json}"

VLLM_BASE_URL="${VLLM_BASE_URL:-http://127.0.0.1:8081/v1}"
N_SAMPLES="${N_SAMPLES:-5}"
Y1_N="${Y1_N:-1}"
MAX_ROWS="${MAX_ROWS:--1}"
GPT_VERIFY_MODEL="${GPT_VERIFY_MODEL:-${GPT_CRITIQUE_MODEL:-gpt-5.6-terra}}"
KEEP_PROB_SCALE="${KEEP_PROB_SCALE:-1.0}"
SEED="${SEED:-42}"

# comma list: seed,solutions,verify,y1,construct,convert
STAGES="${STAGES:-seed,solutions,verify,y1,construct,convert}"
SKIP_VERIFY="${SKIP_VERIFY:-0}"
SKIP_Y1="${SKIP_Y1:-0}"

has_stage() {
  [[ ",$STAGES," == *",$1,"* ]]
}

echo "[collect] tag=$SFT_TAG OUT_DIR=$OUT_DIR PAG_OUT=$PAG_OUT STAGES=$STAGES"

if has_stage seed; then
  echo "[$(date)] Stage seed"
  "$PY" "$SFT_DIR/prepare_seed_jsonl.py" \
    --input "$SEED_PARQUET" \
    --output "$SEED_FILE" \
    --max_rows "$MAX_ROWS"
fi

if has_stage solutions; then
  echo "[$(date)] Stage solutions (model=$SOLUTION_MODEL url=$VLLM_BASE_URL)"
  "$PY" "$SFT_DIR/collect_solutions.py" \
    --seed_file "$SEED_FILE" \
    --output "$SOLUTIONS" \
    --base_url "$VLLM_BASE_URL" \
    --model "$SOLUTION_MODEL" \
    --n "$N_SAMPLES" \
    --max_rows "$MAX_ROWS"
fi

if has_stage verify; then
  if [[ "$SKIP_VERIFY" == "1" ]]; then
    echo "[$(date)] Stage verify SKIPPED (SKIP_VERIFY=1)"
  else
    if [[ -z "${OPENAI_API_KEY:-}" ]]; then
      echo "OPENAI_API_KEY required for verify stage" >&2
      exit 1
    fi
    echo "[$(date)] Stage verify (model=$GPT_VERIFY_MODEL)"
    "$PY" "$SFT_DIR/collect_verifications.py" \
      --solutions "$SOLUTIONS" \
      --output "$VERIFICATIONS" \
      --model "$GPT_VERIFY_MODEL" \
      --max_rows "$MAX_ROWS"
  fi
fi

if has_stage y1; then
  if [[ "$SKIP_Y1" == "1" ]]; then
    echo "[$(date)] Stage y1 SKIPPED (SKIP_Y1=1)"
  else
    echo "[$(date)] Stage y1 (base model + GPT critique, n=$Y1_N)"
    "$PY" "$SFT_DIR/collect_y1.py" \
      --verifications "$VERIFICATIONS" \
      --output "$Y1" \
      --base_url "$VLLM_BASE_URL" \
      --model "$SOLUTION_MODEL" \
      --n "$Y1_N" \
      --max_rows "$MAX_ROWS"
  fi
fi

if has_stage construct; then
  echo "[$(date)] Stage construct"
  CONSTRUCT_ARGS=(
    --solutions "$SOLUTIONS"
    --verifications "$VERIFICATIONS"
    --output "$S2R_JSON"
    --model "$SOLUTION_MODEL"
    --seed "$SEED"
    --keep_prob_scale "$KEEP_PROB_SCALE"
  )
  if [[ -f "$Y1" ]]; then
    CONSTRUCT_ARGS+=(--y1 "$Y1")
  fi
  "$PY" "$SFT_DIR/construct_s2r_sft.py" "${CONSTRUCT_ARGS[@]}"
fi

if has_stage convert; then
  echo "[$(date)] Stage convert → PAG multi-turn"
  "$PY" "$SFT_DIR/convert_s2r_to_pag_multiturn.py" \
    --input "$S2R_JSON" \
    --out_dir "$PAG_OUT" \
    --seed "$SEED"
fi

echo "[$(date)] Done."
echo "  S2R JSON : $S2R_JSON"
echo "  PAG SFT  : $PAG_OUT/sft_{verify,rectify,mixed}_{train,val}.parquet"
