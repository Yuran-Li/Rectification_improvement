<h1 style="text-align: center;">PAG: Multi-Turn Reinforced LLM Self-Correction with Policy as Generative Verifier</h1>

<div align="center">

[![Paper](https://img.shields.io/badge/paper-A42C25?style=for-the-badge&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2506.10406)
[![HomePage](https://img.shields.io/badge/home-000000?style=for-the-badge&logo=homeassistant&logoColor=white)](https://jackory.github.io/pag/)

</div>

## News
- **[2026/08/18]** Critique-informativeness PPO: split verifier rewards, same-`y0` generic counterfactual, pair logging (`P(CW)`, CER)
- **[2025/06/27]** 🎉 Code released
- **[2025/06/13]** 🎉 [HomePage](https://jackory.github.io/pag/) released

## Installation
This repository is based on verl commit 81a15ed7 (2025/04/03) and requires FSDP with vLLM **0.8.2** (see upstream [verl install](https://verl.readthedocs.io/en/latest/start/install.html)). Also install [Math-Verify](https://github.com/huggingface/Math-Verify): `pip install math-verify`.

**Local reproduction (recommended):** follow [`docs/ENV.md`](docs/ENV.md). File roles:

| File | Role |
|------|------|
| `requirements.txt` | Declared deps (install entry) |
| `environment.yml` | Conda skeleton + critical pins |
| `requirements.freeze.txt` | Known-good `pip freeze` snapshot (**reference only**) |
| `docker/Dockerfile.ngc.vllm0.8` | Container baseline |

Critical pins used in our runs: Python 3.10, `torch==2.6.0`, `vllm==0.8.2`, `flash-attn==2.7.4.post1` (cu12/torch2.6 cxx11abiFALSE wheel).

## Quick Start
We provide training scripts for PAG and baseline methods including [SCoRe](https://arxiv.org/pdf/2409.12917) and Direct_MultiTurn:

- PAG (upstream-style): `bash quick_start/qwen1p5b_pag.sh`
- PAG (local paths / flags): `bash quick_start/run_pag_local.sh`  
  Activate the `PAG` conda env first. Default is **1.5B + MATH 7.5k**. Override with `MODEL_PATH`, `N_GPUS`, `HF_HOME`, `USE_WANDB=1`. See [current training changes](#current-training-changes) below.
- Eval (local): `bash quick_start/run_eval_local.sh`
- SCoRe: `bash quick_start/qwen1p5b_SCoRe.sh` 
- Direct_MultiTurn: `bash quick_start/qwen1p5b_multiturn.sh`

The evaluation pipeline follows the same procedure as training, please refer to `quick_start/evaluation.sh` for more details.

For debugging purposes, we provide two multi-turn test scripts:
- `tests/multi_turn/run_vllm_spmd_pag_rollout.py`
- `tests/multi_turn/run_vllm_spmd_direct_multiturn.py`

If you encounter CUDA errors during debugging, try commenting out `self.inference_engine.sleep(level=1)` in:
- `verl/workers/rollout/vllm_rollout/vllm_pag_rollout_spmd.py`
- `verl/workers/rollout/vllm_rollout/vllm_multiturn_rollout_spmd.py`

Note that this is only for debugging purposes.

## Current training changes

These changes sit on top of the original PAG loop (`n=4` independent `y0` samples per prompt). They are **on** in `quick_start/run_pag_local.sh` and `quick_start/qwen1p5b_pag.sh`.

### What changed

1. **Verifier user prompt** (`verl/utils/pag_prompts.py`): one generation that writes a short error critique, then a hard closer `The answer is wrong.` / `The answer is correct.`. No full re-solve and no `\boxed{}` in verify.
2. **Split verifier rewards** (`reward_model.split_verify_reward=True`):
   - `R_disc` = GenRM score on the last **verdict** token.
   - Usefulness on the last **feedback** token (see `R_critique` below).
   - GAE uses `γ=1`, `λ=1`. Verdict-last placement credits feedback+verdict; feedback-last placement does **not** flow into the rectifier.
3. **Generic counterfactual fork** (`actor_rollout_ref.rollout.generic_counterfactual=True`): if a row would already rectify (same `should_revise` gate as PAG, default = verifier says `wrong`), sample one extra recovery from **that same** `y0`:

   `x → y0 → VERIFY_USER → v_generic → REGENERATE_USER → y_generic`

   `v_generic` is the fixed string `GENERIC_CRITIQUE` (format-matched: non-specific feedback + `The answer is wrong.`). Do **not** resample `y0`. Do **not** pair across rollout indices. `n` stays 4; this is not `n=8`.

4. **`y_generic` is not a PPO sequence.** It is stored on the self row (`generic_response`, `traj_id`) and used only to score the contrast. Actor/critic still see only `x → y0 → v_self → y_self`. `include_generic_in_actor=False` (True is not implemented).

### Rewards on the self row

| Signal | Token | Formula |
|--------|--------|---------|
| `R_disc` | last verdict token | GenRM score (correct/wrong vs GT) |
| `R_critique` | last self-feedback token | `R_y(y_self) - R_y(y_generic)` in `{+1, 0, -1}` |
| `R_rect` | last rectifier token | `R_y(y_self) + rs_coef * Δ_self` (`policy_rs` unchanged) |

No rectify ⇒ no generic fork ⇒ no `R_critique`. User-template gaps (`multiturn_mask=False`) still zero GAE into `y0`.

### Pair logging (train step + full val window)

Among **forked** rows only, label `(self, generic)`:

| Cell | Meaning | `R_cf` |
|------|---------|--------|
| CW | self correct, generic wrong | +1 |
| CC | both correct | 0 |
| WW | both wrong | 0 |
| WC | self wrong, generic correct | −1 |

Logged as `multiturn/*` (train) and `val/multiturn/*` (aggregated over the whole validation pass):

- `p_cw`, `p_cc`, `p_ww`, `p_wc`
- `e_r_cf` = `P(CW) - P(WC)`
- `critique_exclusive_rate` = `CW / (CW + CC)` (omitted if self never succeeds)

The method target is **`P(CW)` ↑** (specific critique adds corrective information), not only self accuracy. Rising `p_cc` with falling CER usually means the rectifier got stronger, not that critiques got more informative.

### Launch

```bash
conda activate PAG
export HF_HOME=/path/to/hf-cache   # on this machine: /data/yuranli/hf-cache
cd /path/to/project-new-method
N_GPUS=8 bash quick_start/run_pag_local.sh
```

`run_pag_local.sh` picks the train set from `MODEL_PATH`:

| Model | Train parquet | Val |
|-------|----------------|-----|
| 1.5B (default `Qwen/Qwen2.5-1.5B-Instruct`) | `datasets/math7500.parquet` | `math500` |
| 7B (`MODEL_PATH=...7B...`) | `datasets/dapo17k.parquet` | `math500` |

Override with `TRAIN_DATASET=/path/to.parquet`. Other flags: `USE_WANDB=1`, `EXPERIMENT_NAME=...`, `GPU_MEM_UTIL=0.6`, `CUDA_VISIBLE_DEVICES=...`.

Unit tests: `PYTHONPATH=. python tests/test_split_verify_reward.py` (PAG env).

## Citation
If you find this project helpful, please cite:

```bibtex
@article{jiang2025pag,
  title={PAG: Multi-Turn Reinforced LLM Self-Correction with Policy as Generative Verifier},
  author={Jiang, Yuhua and Xiong, Yuwen and Yuan, Yufeng and Xin, Chao and Xu, Wenyuan and Yue, Yu and Zhao, Qianchuan and Yan, Lin},
  journal={arXiv preprint arXiv:2506.10406},
  year={2025}
}