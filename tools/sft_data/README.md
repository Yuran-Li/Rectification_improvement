# SFT data (S2R → PAG multi-turn)

Convert official S2R SFT (`sft_qwen2.5_math_7B.json`) into PAG-style multi-turn
`messages` for `verl.utils.dataset.MultiTurnSFTDataset`.

## Convert

```bash
cd /data/yuranli/LLM/2026.04/ICLR_2027/Rectification_improvement

python tools/sft_data/convert_s2r_to_pag_multiturn.py \
  --input /data/yuranli/LLM/2026.04/github_references/S2R/data/train_data/sft_qwen2.5_math_7B.json \
  --out_dir datasets/sft
```

## Outputs

| File | Meaning |
|------|---------|
| `sft_verify_{train,val}.parquet` | Loss on **verify** turn only (~3.2k) |
| `sft_rectify_{train,val}.parquet` | Loss on **rectify** turn only (~1.6k wrong→revise) |
| `sft_mixed_{train,val}.parquet` | verify + rectify rows concat (joint warmup) |
| `parsed_s2r.jsonl` | Intermediate fields (`y0` / `verify` / `rectify`) |
| `conversion_stats.json` | Counts / errors |

User templates match `vllm_pag_rollout_spmd.py` (`VERIFY_TEMPLATE` / `REGENERATE_TEMPLATE`),
including the confirmative constraint: **verify without re-solving from scratch**.
Closing verdicts are normalized to `The answer is wrong.` / `The answer is correct.`
Rectify targets keep only the final revision (nested S2R Wait/retry loops stripped).

## Train

`MultiTurnSFTDataset` only supervises the **last** assistant message, so use the
split files (or `mixed`) rather than one row with both turns trainable.

```bash
# verify → rectify sequential (recommended)
EVAL_EVERY=20 EVAL_N_PROBLEMS=32 N_GPUS=6 \
  bash quick_start/run_sft_verify_then_rectify.sh
```

During SFT, every `EVAL_EVERY` steps the trainer runs PAG multi-turn generate on a
MATH500 subset and logs to wandb:

- `val/pag/TPR`, `val/pag/TNR`
- `val/pag/ECR_TP`, `val/pag/EIR_FP`
- `val/pag/verify_acc`, `val/pag/a1_acc`, `val/pag/final_acc`
- `val/pag/i_to_c_rate`, `val/pag/c_to_i_rate`

Set `EVAL_EVERY=0` to disable. Note: HF generate eval is slow; 20 steps × 32 problems is a reasonable default.
