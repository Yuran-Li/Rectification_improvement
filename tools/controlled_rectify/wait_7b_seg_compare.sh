#!/usr/bin/env bash
# Wait for 7B SCOPE/STRIDE pools, then teacher + eval + compare.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
CR=tools/controlled_rectify
POOL_SCOPE=$CR/data/fixed_wrong_qwen7b_instruct_scope.jsonl
POOL_STRIDE=$CR/data/fixed_wrong_qwen7b_instruct_stride.jsonl

echo "[$(date)] waiting for pools"
while true; do
  if [[ -f "$POOL_SCOPE" && -f "$POOL_STRIDE" ]] && ! pgrep -f 'build_prerl_pool.py' >/dev/null; then
    break
  fi
  echo "[$(date)] still waiting pools; procs=$(pgrep -cf 'build_prerl_pool.py' || true)"
  sleep 60
done
echo "[$(date)] pools ready: $(wc -l < "$POOL_SCOPE") scope / $(wc -l < "$POOL_STRIDE") stride"

bash "$CR/run_7b_seg_compare.sh" teachers
bash "$CR/run_7b_seg_compare.sh" evals
echo "[$(date)] 7b seg compare done"
