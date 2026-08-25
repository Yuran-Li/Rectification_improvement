# Controlled Rectification Eval (PAG)

Fill the controlled-rectification table for **PAG Pre-RL** vs **PPO (global_step_400)** on MATH-500.

| Column | Meaning |
|--------|---------|
| \(\mathrm{ECR}_{\mathrm{gen}}\) | PAG turns: \(y_0\) + generic verify (“… The answer is wrong.”) + regenerate user |
| \(\mathrm{ECR}_{\mathrm{fix}}\) | same turns, but verify body = **shared GPT critique** (+ forced wrong close) |
| \(\mathrm{Acc}_{\mathrm{regen}}\) | problem only (no \(y_0\) / verify / regenerate) |
| \(\Delta_{\mathrm{rect}}^{@1}\) | \(\mathrm{ECR}_{\mathrm{fix}}^{@1}-\mathrm{Acc}_{\mathrm{regen}}^{@1}\) |
| \(\Delta_{\mathrm{base}}^{@1}\) | \(\mathrm{ECR}_{\mathrm{fix}}^{@1}\) vs Pre-RL |

## Protocol

1. **Oracle \(a_1\)**: GT grading only (`extract_answer` + `math_equal`). Wrong ⇒ enter pool. No GenRM / no FP/FN from verify.
2. **Fixed pool**: Pre-RL (`Qwen2.5-1.5B-Instruct`) one \(y_0\) per MATH-500 problem; same pool for Pre-RL & PPO.
3. **One-round revise only**, prompts match **PAG RL ChatML** (not S2R bridges):
   `problem → y0 → VERIFY_USER → verify body (… The answer is wrong.) → REGENERATE_USER → y1`.
4. **Fixed critiques**: **GPT API** (`generate_fixed_critiques_gpt.py`); injected as the verify assistant turn and forced to end with `The answer is wrong.`
5. **@1 / @8**: `T=0,n=1` vs `T=0.7,n=8` (pass@8).

## Layout

```
tools/controlled_rectify/
  build_prerl_pool.py
  generate_fixed_critiques_gpt.py   # GPT API → fixed_critique
  controlled_rectify_eval.py
  aggregate_table.py
  run_pipeline.sh
  data/
  results/
```

## Setup

```bash
export OPENAI_API_KEY=...
# optional:
export OPENAI_BASE_URL=...          # Azure / proxy
export GPT_CRITIQUE_MODEL=gpt-5o    # or your GPT-5.x id
export CUDA_VISIBLE_DEVICES=0,1
export TP=2
```

PPO FSDP shards must be merged once:

```bash
bash tools/controlled_rectify/run_pipeline.sh merge
# → checkpoints/PAG/qwen1p5b_pag/global_step_400/actor_hf
```

## Run

```bash
cd /path/to/Policy-As-GenVerifier

# step by step
bash tools/controlled_rectify/run_pipeline.sh pool
bash tools/controlled_rectify/run_pipeline.sh critique   # needs OPENAI_API_KEY
bash tools/controlled_rectify/run_pipeline.sh eval_at1
bash tools/controlled_rectify/run_pipeline.sh eval_at8
bash tools/controlled_rectify/run_pipeline.sh aggregate

# or everything (needs free GPUs + API key)
bash tools/controlled_rectify/run_pipeline.sh all
```

Outputs:

- `results/pag_prerl_at{1,8}.metrics.json`
- `results/pag_ppo400_at{1,8}.metrics.json`
- `results/table_pag_controlled_rectify.tex`

## Diagnose Reward Informativeness

Goal: for each fixed wrong `(problem, y0)`, sample `M` verify texts (specific critiques),
then under each verify sample `K` regenerate outcomes; decompose variance into:

- between-critique utility variation: `Var_j(p_j)`, where `p_j = mean_k a2(j,k)`
- within-critique outcome variance: `mean_j p_j(1-p_j)`
- `rho = Var_between / (Var_between + Var_within)`

Also quantify GRPO group composition (all-zero / all-one / informative mixed) and
compare 4 conditions on the same pool/rectifier/temperature:

- specific critique (sampled verifies)
- generic feedback
- no feedback (`The answer is wrong.` only)
- independent re-solving (problem only)

`sample_verifies_multi.py` writes both:
- `verifies_raw`: model output as sampled
- `verifies_for_regen`: strip verdict + force `The answer is wrong.` (used for regenerate)
- `verifies_verdict_raw`: parsed raw verdict per sample (`correct` / `wrong` / `none`)

Run:

```bash
# defaults diagnose PPO-400 model for both verify and rectify
export DIAG_TAG=pag_ppo400
export DIAG_M=8
export DIAG_K=8
export DIAG_TEMP=0.7
# optional override:
# export DIAG_VERIFY_MODEL=$PPO_HF
# export DIAG_RECTIFIER_MODEL=$PPO_HF

bash tools/controlled_rectify/run_pipeline.sh diag_sample_verify
bash tools/controlled_rectify/run_pipeline.sh diag_informativeness
bash tools/controlled_rectify/run_pipeline.sh diag_aggregate
# or one-shot:
bash tools/controlled_rectify/run_pipeline.sh diag_all
```

Outputs:

- `results/reward_informativeness_<tag>_M<M>_K<K>.jsonl`
- `results/reward_informativeness_<tag>_M<M>_K<K>.summary.json`
- `results/reward_informativeness_<tag>_M<M>_K<K>_tables.tex`

## Causal rectify eval (feedback × edit constraint)

Clean ablation **before joint RL**: same wrong pool / same rectifier / same budget;
only change feedback content and whether editing must preserve the correct prefix.

Feedback axis: regenerate / wrong_only / localization / loc+analysis / loc+analysis+plan / freeform  
Edit axis: full_regen vs prefix_rewrite  

```bash
# Teacher structured feedback v=(c,t*,e,p)
# Prefer GPT if OPENAI_API_KEY is set; otherwise falls back to local PPO HF as teacher.
export CAUSAL_BACKEND=auto   # or gpt / local
export TP=1
CUDA_VISIBLE_DEVICES=0 bash tools/controlled_rectify/run_pipeline.sh causal_teacher

# 2D matrix (@1 by default)
CUDA_VISIBLE_DEVICES=0 bash tools/controlled_rectify/run_pipeline.sh causal_eval
bash tools/controlled_rectify/run_pipeline.sh causal_aggregate
# or one-shot:
CUDA_VISIBLE_DEVICES=0 bash tools/controlled_rectify/run_pipeline.sh causal_all
```

Outputs:
- `data/fixed_wrong_pag_prerl_teacher_structured.jsonl`
- `results/causal_rectify_pag_ppo400.jsonl`
- `results/causal_rectify_pag_ppo400.summary.json` / `_tables.tex`

Reported metrics: W2C, PPR, FCR, \(\Delta_{\mathrm{critique}}\) (true vs shuffled plan).

## Note on GPUs / vLLM

Pool + eval need free GPUs. Critique (GPT) is **API-only** and can run while GPUs are busy.

Defaults match PAG train/eval to avoid V1 sampler OOM:

- `VLLM_USE_V1=0` (forced in `run_pipeline.sh`)
- `GPU_MEM_UTIL=0.70`, `MAX_NUM_SEQS=256` (override via env if needed)
