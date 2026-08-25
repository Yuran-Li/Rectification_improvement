# SFT data for PAG (S2R-style collect → PAG multi-turn)

Two ways to build PAG multi-turn SFT:

1. **Convert release S2R JSON** (fast, no GPT) — official `sft_qwen2.5_math_7B.json`
2. **Collect your own** (for 1.5B / new corpora) — migrated S2R pipeline under this folder

Final train files are parquet consumed by `verl.utils.dataset.MultiTurnSFTDataset`
via `quick_start/run_sft_pag_multiturn.sh` / `run_sft_verify_then_rectify.sh`.

---

## A. Convert existing S2R SFT (no API)

```bash
cd /data/yuranli/LLM/2026.04/ICLR_2027/project-new-method

python tools/sft_data/convert_s2r_to_pag_multiturn.py \
  --input /data/yuranli/LLM/2026.04/github_references/S2R/data/train_data/sft_qwen2.5_math_7B.json \
  --out_dir datasets/sft/Qwen-7B
```

---

## B. Collect S2R-style data yourself (migrated pipeline)

Upstream S2R flow (`1_collect → 2_verify(GPT) → 3_construct`) lives here as:

| Stage | Script | Needs |
|-------|--------|--------|
| 0 seed | `prepare_seed_jsonl.py` | parquet/jsonl → seed |
| 1 solutions | `collect_solutions.py` | **local vLLM** → y0 |
| 2 verify | `collect_verifications.py` | **GPT API** |
| 2b y1 | `collect_y1.py` | **local vLLM** + GPT critique → y1（不加 GPT 费） |
| 3 construct | `construct_s2r_sft.py` | 正确尾部从 **gold y0 ∪ gold y1** 抽 |
| 4 convert | `convert_s2r_to_pag_multiturn.py` | → PAG parquet |

Stitch is still S2R-style (`y_wrong + Wait + verify + Let me try again + y_correct`).
`y_correct` prefers a **paired gold y1** (same wrong y0 that the critique addressed); if y1 is still wrong, fall back to a gold y0. Problems with no gold y0 but a gold y1 are kept (rescued).

### One-shot

```bash
# terminal A: serve policy used for y0 sampling
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-1.5B-Instruct --port 8081

# terminal B
conda activate PAG   # or your project env with openai/transformers/pandas
export OPENAI_API_KEY=...
# optional: export OPENAI_BASE_URL=...
# optional: export GPT_VERIFY_MODEL=gpt-5.6-terra

cd /data/yuranli/LLM/2026.04/ICLR_2027/project-new-method

# smoke
MAX_ROWS=50 bash tools/sft_data/run_collect_pipeline.sh

# full MATH-7.5k with 1.5B
SOLUTION_MODEL=Qwen/Qwen2.5-1.5B-Instruct \
  SEED_PARQUET=datasets/math7500.parquet \
  bash tools/sft_data/run_collect_pipeline.sh
```

Writes `datasets/sft_collect/Qwen-1.5B/` (raw) and `datasets/sft/Qwen-1.5B/` (PAG parquet). Override with `SFT_TAG` / `OUT_DIR` / `PAG_OUT`.

Outputs:

| Path | Role |
|------|------|
| `datasets/sft_collect/Qwen-1.5B/seed.jsonl` | prompts |
| `datasets/sft_collect/Qwen-1.5B/solutions.jsonl` | n× y0 per problem |
| `datasets/sft_collect/Qwen-1.5B/verifications.jsonl` | GPT critiques |
| `datasets/sft_collect/Qwen-1.5B/y1.jsonl` | base-model revisions given GPT critique |
| `datasets/sft_collect/Qwen-1.5B/sft_s2r_style.json` | S2R monologue (same schema as release) |
| `datasets/sft/Qwen-1.5B/sft_{verify,rectify,mixed}_{train,val}.parquet` | PAG SFT (1.5B) |
| `datasets/sft/Qwen-7B/sft_{verify,rectify,mixed}_{train,val}.parquet` | PAG SFT (official 7B convert) |

### Stage control

```bash
# only seed + solutions
STAGES=seed,solutions MAX_ROWS=100 bash tools/sft_data/run_collect_pipeline.sh

# resume verify after solutions exist
STAGES=verify,y1,construct,convert bash tools/sft_data/run_collect_pipeline.sh

# reuse existing verifications; still (re)sample y1
SKIP_VERIFY=1 STAGES=y1,construct,convert bash tools/sft_data/run_collect_pipeline.sh

# stitch y0-only (official S2R, no y1)
SKIP_Y1=1 STAGES=construct,convert bash tools/sft_data/run_collect_pipeline.sh

# only convert a finished S2R JSON
STAGES=convert S2R_JSON=datasets/sft_collect/Qwen-1.5B/sft_s2r_style.json \
  PAG_OUT=datasets/sft/Qwen-1.5B \
  bash tools/sft_data/run_collect_pipeline.sh
```

Env knobs: `VLLM_BASE_URL`, `SOLUTION_MODEL`, `N_SAMPLES` (default 5),
`Y1_N` (default 1 y1 per unique wrong y0), `GPT_VERIFY_MODEL`,
`KEEP_PROB_SCALE`, `OUT_DIR`, `PAG_OUT`.

Steps 1–2b are **append / resume-safe** (`unique_id`, or `(unique_id, answer)` for y1).

---

## Convert outputs (shared)

| File | Meaning |
|------|---------|
| `sft_verify_{train,val}.parquet` | Loss on **verify** turn only |
| `sft_rectify_{train,val}.parquet` | Loss on **rectify** turn only (wrong→revise) |
| `sft_mixed_{train,val}.parquet` | verify + rectify concat |
| `parsed_s2r.jsonl` | Intermediate `y0` / `verify` / `rectify` |
| `conversion_stats.json` | Counts / errors |

User templates match `vllm_pag_rollout_spmd.py` (`VERIFY_TEMPLATE` / `REGENERATE_TEMPLATE`),
including the confirmative constraint: **verify without re-solving from scratch**.
Closing verdicts are normalized to `The answer is wrong.` / `The answer is correct.`
Rectify targets keep only the final revision (nested S2R Wait/retry loops stripped).

---

## Train

`MultiTurnSFTDataset` only supervises the **last** assistant message, so use the
split files (or `mixed`) rather than one row with both turns trainable.

```bash
# 1.5B (default: datasets/sft/Qwen-1.5B)
SPLIT=mixed N_GPUS=8 MODEL_PATH=Qwen/Qwen2.5-1.5B-Instruct \
  bash quick_start/run_sft_pag_multiturn.sh

# 7B (auto-picks datasets/sft/Qwen-7B from MODEL_PATH)
SPLIT=mixed N_GPUS=8 MODEL_PATH=Qwen/Qwen2.5-7B-Instruct \
  bash quick_start/run_sft_pag_multiturn.sh
```

SFT adds KL to the frozen Instruct init (`optim.lm_kl_coeff=0.01`, same as S2R 7B).
Verify-only SFT dropped MATH-500 A1 50.6→39.4; KL is on **all non-pad tokens** so y0
is regularized even though CE only hits the last turn. Disable with `LM_KL_COEFF=0`.
Do **not** run sequential rectify-only after verify-only (it wipes the closer).

---

## Notes vs upstream S2R

- Official release only ships **7B** SFT JSON; there is **no** separate 1.5B SFT dump.
  Collecting with `SOLUTION_MODEL=...1.5B...` is the intended way to get 1.5B-aligned y0s.
- Step 2 still needs a strong teacher API (GPT-class). You can point `--base_url` at any
  OpenAI-compatible endpoint (including a local 72B) if you prefer not to call GPT.
- `construct_s2r_sft.py` drops upstream `breakpoint()` / `split_data` chunking; PAG only
  needs the full monologue JSON that `convert_s2r_to_pag_multiturn.py` already understands.
- Official S2R only stitches a gold **y0**. Here gold **y1** (critique-conditioned base
  retry) also enters the correct pool; convert still takes the text after
  `Let me try again` as the rectify target.
