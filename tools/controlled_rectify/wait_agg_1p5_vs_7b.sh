#!/usr/bin/env bash
# Wait for parallel 1.5B/7B Instruct causal evals, then aggregate + compare.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
PY="${PAG_PY:-/data/yuranli/envs/PAG/bin/python}"
OUT15=tools/controlled_rectify/results/causal_rectify_qwen1p5b_instruct.jsonl
OUT7=tools/controlled_rectify/results/causal_rectify_qwen7b_instruct.jsonl

echo "[$(date)] waiting for $OUT15 and $OUT7"
while true; do
  if [[ -f "$OUT15" && -f "$OUT7" ]] && ! pgrep -f 'tools/controlled_rectify/causal_rectify_eval.py' >/dev/null; then
    break
  fi
  n=$(pgrep -cf 'tools/controlled_rectify/causal_rectify_eval.py' || true)
  echo "[$(date)] still waiting; eval_procs=${n:-0}"
  sleep 60
done

echo "[$(date)] aggregating"
"$PY" tools/controlled_rectify/aggregate_causal_rectify.py \
  --input "$OUT15" \
  --out_prefix tools/controlled_rectify/results/causal_rectify_qwen1p5b_instruct
"$PY" tools/controlled_rectify/aggregate_causal_rectify.py \
  --input "$OUT7" \
  --out_prefix tools/controlled_rectify/results/causal_rectify_qwen7b_instruct

"$PY" - <<'PY'
import json
from pathlib import Path

for tag in ["qwen1p5b_instruct", "qwen7b_instruct", "pag_ppo400"]:
    p = Path(f"tools/controlled_rectify/results/causal_rectify_{tag}.summary.json")
    if not p.exists():
        print(tag, "missing")
        continue
    s = json.loads(p.read_text())
    m = s["metrics"]
    print(f"\n=== {tag} n={s['n_rows']} delta_critique={s.get('delta_critique_W2C')} ===")
    for fb in [
        "regenerate",
        "wrong_only",
        "localization",
        "localization_analysis",
        "localization_analysis_plan",
        "freeform",
    ]:
        fr = m.get(f"{fb}__full_regen", {}).get("W2C@1_pct")
        pr = m.get(f"{fb}__prefix_rewrite", {}).get("W2C@1_pct")
        print(f"  {fb:28s} full={fr}  prefix={pr}")
PY
echo "[$(date)] done"
