<h1 style="text-align: center;">PAG: Multi-Turn Reinforced LLM Self-Correction with Policy as Generative Verifier</h1>

<div align="center">

[![Paper](https://img.shields.io/badge/paper-A42C25?style=for-the-badge&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2506.10406)
[![HomePage](https://img.shields.io/badge/home-000000?style=for-the-badge&logo=homeassistant&logoColor=white)](https://jackory.github.io/pag/)

</div>

## News
- **[2026/08/28]** Dynamic-category SEC: C1–C5 from synchronized $(g_t, n_{WC,t})$ refreshes (this branch). Q-update and sampling are unchanged from fixed MATH-level SEC.
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

An optional online curriculum sampler that replaces uniform prompt sampling with a dynamic C1–C5 bandit. SEC and the previous generation-frontier curriculum are **mutually exclusive**.

### Design

SEC maintains a Q-value per **dynamic** category C1–C5 and updates it after every PPO step using the generation-turn advantage as the reward signal. Category membership is **not** the parquet MATH level; it is assigned from a synchronized full-train PAG measurement of the current policy:

```
C1: g = 1
C2: 0.5 ≤ g < 1, n_WC > 0
C3: 0.5 ≤ g < 1, n_WC = 0
C4: g < 0.5,     n_WC > 0
C5: g < 0.5,     n_WC = 0
```

`g = n_correct_y0 / K_refresh` and `n_WC = #{y0 wrong and rectified y2 correct}`, using the same PAG `acc >= 0.5` criteria as training. Per-prompt state is keyed by `extra_info.index`.

```
Q_0(c) = 0  for all c
P_t(c) = softmax(Q_t / τ)   # empty categories masked

for each batch of size B:
    for each slot b = 1..B (i.i.d., with replacement):
        c_b ~ Categorical(P_t)
        prompt_b ~ Uniform(C_{c_b})   # with replacement

    → normal PAG rollout + advantage computation
    → r_t(c) = mean u_i^G over prompts i from category c
    → Q_{t+1}(c) = (1-α) Q_t(c) + α r_t(c)   (absent categories unchanged)
    → P_{t+1}
```

Every `refresh_interval` PPO steps the trainer pauses for a **measurement** (not a uniform-training epoch): full-train PAG rollout of the current actor (FSDP weights synced into vLLM), then an atomic replace of `prompt_id → category`. Q is not reset.

`u_i^G` is the mean absolute generation-turn (`y0`) advantage over the K rollout trajectories of prompt `i`:

```
u_i^G = (1/K) Σ_k (1/T_ik^G) Σ_{t ∈ y0,ik} |A^G_{ik,t}|
```

Only `A^G` (generation advantage) feeds the Q update. Verification and rectification advantages are logged per category for diagnostics only (`sec/A_verify_C*`, `sec/A_rectify_C*`).

### Dynamic categories

There is no U→C→U→C schedule. SEC stays on. Refreshes are synchronized: every prompt is measured from the same policy snapshot. Precomputed `initial_category_stats_path` JSON must embed a `protocol` block matching this run (`refresh_rollouts` / model / sampling / PAG `revise_gate`). **Do not load K=8 dumps into a K=4 run.**

### DataLoader bypass

Because `Q_{t+1}` must determine batch `t+1`, SEC bypasses `StatefulDataLoader` entirely when enabled. The trainer calls `sec_sampler.sample_batch(B)` to get indices, then fetches and collates items from `train_dataset` on the main process before dispatching to workers. No prefetching with stale Q-values.

### Configuration

```yaml
# ppo_trainer.yaml
sec:
  enabled: false
  q_alpha: 0.1
  temperature: 1.0
  log_path: null
  refresh_interval: 50
  refresh_rollouts: 4          # must equal actor_rollout_ref.rollout.n
  initial_category_stats_path: null
```

`sec.enabled` and `curriculum.enabled` cannot both be `true` (assertion in trainer + shell guard).

### Launch

```bash
# SEC training (1.5B, 6 GPUs)
SEC_ENABLED=true SEC_Q_ALPHA=0.1 SEC_TEMPERATURE=1.0 \
  SEC_REFRESH_INTERVAL=50 SEC_REFRESH_ROLLOUTS=4 \
  FEEDBACK_MODE=delta CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 \
  PROJECT_NAME="PAG-sec" EXPERIMENT_NAME="qwen1p5b_pag_sec_dynamic" \
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

Refresh wall-clock (`dynamic/refresh_wall_time_s`) and trajectory counts (`dynamic/refresh_n_trajectories`) are logged separately from training step time. The ~rollout-workload fraction at interval 50 is **not** the same as wall-clock overhead.

### WandB metrics logged per step

| Metric | Description |
|--------|-------------|
| `sec/Q_C{1..5}` | Current Q-values |
| `sec/P_C{1..5}` | Sampling probabilities |
| `sec/reward_C{1..5}` | `r_t(c)` used for Q update |
| `sec/batch_count_C{1..5}` | Number of prompts per category in this batch |
| `sec/cumulative_frac_C{1..5}` | Cumulative fraction of total prompts seen per category |
| `sec/mean_sampled_category` | Mean category id in this batch |
| `sec/A_generate_C{1..5}` | Mean absolute generation advantage per category |
| `sec/A_verify_C{1..5}` | Mean absolute verification advantage (diagnostic) |
| `sec/A_rectify_C{1..5}` | Mean absolute rectification advantage (diagnostic) |

At each synchronized refresh: `dynamic/category_count_C*`, `dynamic/category_frac_C*`, `dynamic/g_mean_C*`, `dynamic/nWC_positive_frac_C*`, `dynamic/transition_Ci_to_Cj`, `dynamic/refresh_wall_time_s`, `dynamic/refresh_n_trajectories`.

### Unit tests

```bash
PYTHONPATH=. python -m pytest tests/test_sec_sampler.py -v
```

Tests cover: C1–C5 assignment, membership keyed by `extra_info.index` (not row position), migration after refresh, Q frozen across reassignment, empty-category mask, protocol-matched stats loading (K=8 dumps rejected on K=4), checkpoint restore of mapping+Q, generation-only `|A_G|`, and no U→C epoch switch.

### Key files

| File | Role |
|------|------|
| `verl/utils/dataset/sec_sampler.py` | `SECSampler` (dynamic membership, sampling, Q update, checkpointing) |
| `verl/trainer/ppo/ray_trainer.py` | Initial/periodic refresh, FSDP→vLLM sync, Q update, logging |
| `verl/trainer/config/ppo_trainer.yaml` | `sec:` config block |
| `quick_start/qwen1p5b_pag.sh` | `SEC_ENABLED`, `SEC_REFRESH_INTERVAL`, `SEC_REFRESH_ROLLOUTS` |
| `quick_start/run_pag_local.sh` | same `sec.*` hydra args (off by default) |
| `tests/test_sec_sampler.py` | Unit tests |

## Citation
If you find this project helpful, please cite:

```bibtex
@article{jiang2025pag,
  title={PAG: Multi-Turn Reinforced LLM Self-Correction with Policy as Generative Verifier},
  author={Jiang, Yuhua and Xiong, Yuwen and Yuan, Yufeng and Xin, Chao and Xu, Wenyuan and Yue, Yu and Zhao, Qianchuan and Yan, Lin},
  journal={arXiv preprint arXiv:2506.10406},
  year={2025}
}