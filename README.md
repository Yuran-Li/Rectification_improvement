<h1 style="text-align: center;">PAG: Multi-Turn Reinforced LLM Self-Correction with Policy as Generative Verifier</h1>

<div align="center">

[![Paper](https://img.shields.io/badge/paper-A42C25?style=for-the-badge&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2506.10406)
[![HomePage](https://img.shields.io/badge/home-000000?style=for-the-badge&logo=homeassistant&logoColor=white)](https://jackory.github.io/pag/)

</div>

## News
- **[2026/09/01]** Prevalence-aware SEC + U/C interleaving (this worktree): \(P(c)\propto |C_c|\exp(Q_c/\tau)\), τ=0.1, no extra measurement refresh. Replaces the 2026/08/28 refresh-interval design.
- **[2026/08/28]** Dynamic-category SEC: C1–C5 from synchronized $(g_t, n_{WC,t})$ refreshes. Q-update and sampling were unchanged from fixed MATH-level SEC.
- **[2026/08/27]** Self-Evolving Curriculum (SEC): online bandit sampler, per-slot multinomial category sampling, generation-advantage Q-update, full WandB diagnostics
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
| `R_critique` (`feedback_mode=generic`) | last self-feedback token | `R_y(y_self) - R_y(y_generic)` in `{+1, 0, -1}` |
| `R_feedback` (`feedback_mode=regen`) | last self-feedback token | `Δ_self * [1 + λ (1 - p_regen)]`; `λ` = `lambda_regen` |
| `R_feedback` (`feedback_mode=delta`) | last self-feedback token | `Δ_self = acc_t2 - acc_t1` ∈ `{+1,0,-1}` (C→C is 0, C→W is −1); generic unused |
| `R_use` (`feedback_mode=acc`) | last self-feedback token | `acc_t2 - λ_cw · 1[C→W]`; `λ_cw` = `lambda_cw` (default 0.2). C→C is `+1`, C→W is `-λ_cw` |
| `R_use` (`feedback_mode=disc`) | — | `0`. Verify is original PAG: only `R_disc` / `genrm_score` at the last verdict token (GAE γ=1 still credits the feedback span). `policy_rs` on y2 is unchanged |
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

Verifier feedback (`reward_model.feedback_mode`, default `generic`):

```bash
# original PAG verify: R_disc only (R_use=0)
FEEDBACK_MODE=disc N_GPUS=8 bash quick_start/run_pag_local.sh
# recommended: acc_t2 with a small C→W penalty (no y_generic fork)
FEEDBACK_MODE=acc LAMBDA_CW=0.2 N_GPUS=8 bash quick_start/run_pag_local.sh
# Δ_self only (no y_generic fork)
FEEDBACK_MODE=delta N_GPUS=8 bash quick_start/run_pag_local.sh
# Δ_self * [1 + λ(1-p_regen)]
FEEDBACK_MODE=regen LAMBDA_REGEN=1.0 N_GPUS=8 bash quick_start/run_pag_local.sh
# R_self - R_generic (samples y_generic)
FEEDBACK_MODE=generic N_GPUS=8 bash quick_start/run_pag_local.sh
# or Hydra:
#   reward_model.feedback_mode=disc
#   reward_model.feedback_mode=acc reward_model.lambda_cw=0.2
#   reward_model.feedback_mode=delta
#   reward_model.feedback_mode=regen reward_model.lambda_regen=1.0
#   reward_model.feedback_mode=generic actor_rollout_ref.rollout.generic_counterfactual=True
```

`FEEDBACK_MODE=generic` turns on the extra `y_generic` rollout. `disc` / `acc` / `regen` / `delta` skip it unless you set `GENERIC_COUNTERFACTUAL=True` (logging only). `lambda_regen` is the regen weight; `lambda_cw` is the C→W penalty for `acc` (not a mode switch).

Unit tests: `PYTHONPATH=. python tests/test_split_verify_reward.py` (PAG env).

---

## Self-Evolving Curriculum (SEC)

An optional online curriculum sampler that replaces uniform prompt sampling with a dynamic C1–C5 bandit. SEC and the previous generation-frontier curriculum (`curriculum.enabled`) are **mutually exclusive**.

This worktree is a **controlled ablation** of the 2026/08/28 dynamic-SEC branch. Only three things change. PAG training, reward, PPO/GAE, C1–C5 thresholds, \(Q=Q^G\), and rollout `n=4` are unchanged.

### What changed vs the previous dynamic SEC

| | Previous (`refresh_interval=50`) | This worktree |
|--|--|--|
| Category sampling | \(P(c)\propto \exp(Q_c/\tau)\) (small cats oversampled) | Switch: `sec.prevalence_aware=true` → \(P(c)\propto \|C_c\|\exp(Q_c/\tau)\); `false` → original softmax |
| Membership refresh | Extra full-train measurement rollout every 50 PPO steps (and at step 0) | **Removed.** Measure while training in U |
| Schedule | Always on the softmax curriculum | `U → C → U → C → …`, start with U |
| Temperature | τ=1.0 | τ=0.1 (still configurable) |
| ε-mix / g(1−g) / multi-role Q | none | still none |

Do **not** pass `sec.refresh_interval`, `sec.refresh_rollouts`, or `SEC_INITIAL_CATEGORY_STATS`. Those keys are gone. Old refresh-interval checkpoints (no `phase` field) are refused.

### Category definitions (unchanged)

```
C1: g = 1
C2: 0.5 ≤ g < 1, n_WC > 0
C3: 0.5 ≤ g < 1, n_WC = 0
C4: g < 0.5,     n_WC > 0
C5: g < 0.5,     n_WC = 0
```

`g = n_correct_y0 / K` with training `K=4`, and `n_WC = #{y0 wrong and rectified y2 correct}` (existing PAG W→C). Per-prompt state is keyed by the stable `extra_info.index`, never filtered-row position.

### 1. Prevalence-aware sampling (switch)

`sec.prevalence_aware` (default `true`; env `SEC_PREVALENCE_AWARE`):

```
# on  (default):
P(c) = |C_c| exp(Q_c / τ) / Σ_j |C_j| exp(Q_j / τ)

# off:
P(c) = exp(Q_c / τ) / Σ_j exp(Q_j / τ)     # empty cats masked in both cases
```

When **on**:

- If all `Q_c` are equal, `P(c) = |C_c| / N` (prompt-uniform).
- A small category is **not** oversampled merely because it is small.
- A category is oversampled relative to its natural prevalence only when its Q is higher.
- Equivalently each prompt has weight `w_i = exp(Q_{c(i)} / τ)`.

When **off**: original category softmax. Equal Q ⇒ uniform over nonempty categories, so a small class still gets `1/n_nonempty` of the slots.

Within a category (C phase): uniform with replacement. Empty categories are masked.

Logged per step: `sec/prevalence_aware`, `sec/category_frac_C*`, `sec/category_prob_C*`, `sec/exposure_multiplier_C*` where `exposure_multiplier_c = P(c) / category_frac(c)`.

### 2. U/C interleaved epochs (no extra measurement rollout)

One U epoch and one C epoch each correspond to about one full training-set pass (`ceil(N/B)` PPO steps). Training starts with a **U** epoch so categories are measured from the current policy without a rollout-only refresh.

**U phase** (uniform training + free state measurement):

- Normal PPO; prompt-uniform, shuffled, **without replacement**; every train prompt once.
- Existing training rollout `n=4`.
- While training, collect per prompt: `prompt_id`, `g`, `n_WC`, generation utility `u_G`.
- Membership stays **frozen** for attribution/logging for the whole U epoch.
- At the **end** of the complete U epoch, atomically: rebuild `prompt_to_category` from the new `(g, n_WC)`; rebuild pools and prevalence; aggregate generation-only utility on the **new** categories; EMA-update Q.

The last U batch may be padded (repeat indices) so `B * n` is divisible by the GPU world size. Padded prompts are **not** double-counted in U statistics.

**C phase** (curriculum training):

- Freeze the membership created at the end of the preceding U epoch.
- Sample from the prevalence-aware distribution; normal PPO.
- Batch-wise Q update using generation-only `|A_G|`.
- Do **not** refresh membership during C.

Then switch back to U.

### 3. Temperature

```yaml
sec:
  temperature: 0.1
```

Configurable, but this experiment uses 0.1. No extra ε/random mixing: U phases already give full prompt-uniform coverage.

### Q semantics (unchanged)

```
Q_c^G ← (1-α) Q_c^G + α r_c^G
```

`r_c^G` is the existing mean generation `|A_G|` utility on `y0`. Do not fold `A_verify` or `A_rectify` into Q. Reward is unchanged.

- **U-end:** one EMA from mean `u_G` of the newly assigned categories.
- **C:** existing batch-wise EMA after each PPO step.

### Checkpoint / resume

`sec_sampler.pt` restores exactly: `Q`, `cumulative_counts`, SEC step, current phase (`U` or `C`), position within the epoch, `prompt_to_category`, `g`, `n_WC`, category pools/prevalence, and RNG state. Resume continues the same U/C phase; it does not restart a U epoch or reset categories.

`total_training_steps` under SEC is `ceil(N/B) * total_epochs` so U can cover the prompts that a `drop_last` DataLoader would skip.

### Configuration

```yaml
# ppo_trainer.yaml
sec:
  enabled: false
  q_alpha: 0.1
  temperature: 0.1
  prevalence_aware: true
  log_path: null
```

`sec.enabled` and `curriculum.enabled` cannot both be `true` (assertion in trainer + shell guard).

### Launch

```bash
# SEC training (1.5B, 6 GPUs)
SEC_ENABLED=true SEC_Q_ALPHA=0.1 SEC_TEMPERATURE=0.1 SEC_PREVALENCE_AWARE=true \
  FEEDBACK_MODE=delta CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 \
  PROJECT_NAME="PAG-sec" EXPERIMENT_NAME="qwen1p5b_pag_sec_prevalence_uc" \
  MODEL_PATH="Qwen/Qwen2.5-1.5B-Instruct" \
  bash quick_start/qwen1p5b_pag.sh \
    trainer.n_gpus_per_node=6 \
    data.train_batch_size=510 \
    actor_rollout_ref.actor.ppo_mini_batch_size=126 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5

# Uniform baseline (SEC off, old curriculum off)
SEC_ENABLED=false CURRICULUM_ENABLED=false \
  FEEDBACK_MODE=delta CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 \
  PROJECT_NAME="PAG-sec" EXPERIMENT_NAME="qwen1p5b_pag_base" \
  MODEL_PATH="Qwen/Qwen2.5-1.5B-Instruct" \
  bash quick_start/qwen1p5b_pag.sh \
    trainer.n_gpus_per_node=6 \
    data.train_batch_size=510 \
    actor_rollout_ref.actor.ppo_mini_batch_size=126 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5
```

### WandB metrics logged per step

| Metric | Description |
|--------|-------------|
| `sec/phase` | 0.0 = U, 1.0 = C |
| `sec/prevalence_aware` | 1.0 if size-weighted P, else 0.0 |
| `sec/epoch_index` | U/C epoch counter |
| `sec/Q_C{1..5}` | Current Q-values |
| `sec/category_prob_C{1..5}` / `sec/P_C{1..5}` | Sampling probabilities |
| `sec/category_frac_C{1..5}` | Natural prevalence \|C_c\|/N |
| `sec/exposure_multiplier_C{1..5}` | P(c) / category_frac(c) |
| `sec/category_count_C{1..5}` | Pool sizes |
| `sec/reward_C{1..5}` | `r_t(c)` used for Q update |
| `sec/batch_count_C{1..5}` | Number of prompts per category in this batch |
| `sec/cumulative_frac_C{1..5}` | Cumulative fraction of total prompts seen per category |
| `sec/A_generate_C{1..5}` | Mean absolute generation advantage per category |
| `sec/A_verify_C{1..5}` | Mean absolute verification advantage (diagnostic) |
| `sec/A_rectify_C{1..5}` | Mean absolute rectification advantage (diagnostic) |

At each U-epoch end: 5×5 `sec/transition_Ci_to_Cj` (and `dynamic/transition_*`), especially C4→C1/C2/C4 and C5→C4/C2/C1.

### Unit tests

```bash
PYTHONPATH=. python -m pytest tests/sec_prevalence_uc_tests.py -v
```

Tests cover: equal Q ⇒ `P(c)=|C_c|/N`; equal-size higher Q ⇒ higher P; per-example exposure ratio \(\exp((Q_i-Q_j)/\tau)\) independent of category size; empty-category mask; U visits every prompt once; membership frozen in U and replaced only at U-end; C uses the new membership; U→C→U→C and checkpoint resume.

### Key files

| File | Role |
|------|------|
| `verl/utils/dataset/sec_sampler.py` | Prevalence-aware `P(c)`, U/C epochs, membership rebuild, Q EMA, checkpoint |
| `verl/trainer/ppo/ray_trainer.py` | U/C batch loop, U recording, no measurement refresh, logging |
| `verl/trainer/config/ppo_trainer.yaml` | `sec.temperature: 0.1`, `sec.prevalence_aware`; no `refresh_interval` |
| `quick_start/qwen1p5b_pag.sh` | `SEC_ENABLED`, `SEC_Q_ALPHA`, `SEC_TEMPERATURE` (default 0.1) |
| `quick_start/run_pag_local.sh` | same `sec.*` hydra args (off by default) |
| `tests/sec_prevalence_uc_tests.py` | Deterministic prevalence / U/C / resume tests |

## Citation
If you find this project helpful, please cite:

```bibtex
@article{jiang2025pag,
  title={PAG: Multi-Turn Reinforced LLM Self-Correction with Policy as Generative Verifier},
  author={Jiang, Yuhua and Xiong, Yuwen and Yuan, Yufeng and Xin, Chao and Xu, Wenyuan and Yue, Yu and Zhao, Qianchuan and Yan, Lin},
  journal={arXiv preprint arXiv:2506.10406},
  year={2025}
}