# Feasibility-Guided Self-Correction

多角色联合自纠错：generate / verify / rectify 共用一个策略，用可行性门控调节 **PG + BC** 加权（不硬关 PG）。

| 模块 | 做法 |
|------|------|
| 轨迹 | 多轮（默认 `num_turns=4`），**滑动窗口**：只保留题目 + 最近一轮答案 |
| 优化 | PPO + GAE；双头 critic \(V_R, V_F\)；Lagrangian CMDP（\(J_F \le B_F\)） |
| Reward | utility-aware：rectify \(R=\hat R+\alpha\Delta\)，verify \(R=\hat R_v+\beta U\) |
| 门控 | \(V_F\le B_F\) → PG + 可选轻量 BC；\(V_F>B_F\) → **保留 PG** + **高权重** expert BC |
| Expert | **同题 bootstrap**（`n` 条里成功段拷给同 `uid` 不可行样本，无 API）；成功段也可轻量 BC。缺口再开 **在线 GPT**（默认关） |

基于 [PAG](https://github.com/Jackory/PAG)（[arXiv:2506.10406](https://arxiv.org/abs/2506.10406)）代码扩展；环境与依赖说明见 [`docs/ENV.md`](docs/ENV.md)。

## Installation

见 [`docs/ENV.md`](docs/ENV.md)。关键 pin：Python 3.10、`torch==2.6.0`、`vllm==0.8.2`、`flash-attn==2.7.4.post1`，并安装 `math-verify`。

## Pipeline

### 1. SFT 预热（verify → rectify）

数据转换与训练说明：[`tools/sft_data/README.md`](tools/sft_data/README.md)

```bash
# 数据
python tools/sft_data/convert_s2r_to_pag_multiturn.py \
  --input /path/to/sft_qwen2.5_math_7B.json \
  --out_dir datasets/sft

# 串行 SFT
EVAL_EVERY=20 EVAL_N_PROBLEMS=32 N_GPUS=6 \
  bash quick_start/run_sft_verify_then_rectify.sh
```

评测候选 ckpt：

```bash
python tools/sft_data/eval_sft_checkpoints.py \
  --math500 datasets/math500.parquet \
  --out_dir results/sft_ckpt_compare
```

默认 RL init：`checkpoints/sft/.../sft_rectify/global_step_75`（若存在）。

### 2. Feasibility-guided RL

```bash
N_GPUS=8 bash quick_start/run_feasibility_pag.sh
```

默认打开：`slide_window`、`utility_aware`、`dual_critic`、`expert_bc`、`bootstrap_same_uid`（**无 API**）。

```bash
# 首版消融（不调 GPT）
ONLINE_GPT_EXPERT=False bash quick_start/run_feasibility_pag.sh

# 最终版：缺口再按需 GPT
export OPENAI_API_KEY=...
ONLINE_GPT_EXPERT=True ONLINE_GPT_MAX_PER_STEP=8 \
  bash quick_start/run_feasibility_pag.sh
```

看日志：`feasibility/bootstrap_transfer_filled`、`actor/expert_bc_loss`。  
常用变量：`NUM_TURNS`、`N_SAMPLES`（建议 ≥4）、`COST_BUDGET`、`EXPERT_BC_COEF`、`EXPERT_BC_LIGHT`。

### 3. 测试

```bash
python tests/test_slide_window_pack.py
```

## 关键代码

| 路径 | 作用 |
|------|------|
| `verl/workers/rollout/vllm_rollout/vllm_pag_rollout_spmd.py` | 滑动窗口多轮 rollout |
| `verl/workers/reward_manager/pag.py` | utility reward / \(c^F\) / expert mask |
| `verl/workers/critic/dp_critic.py`、`fsdp_workers.py` | 双头 \(V_R/V_F\) |
| `verl/trainer/ppo/ray_trainer.py` | Lagrangian、可行性门控、在线 GPT 接入 |
| `verl/trainer/ppo/online_gpt_expert.py` | 按需 GPT expert |
| `verl/trainer/ppo/expert_buffer.py` | bootstrap expert buffer |
| `verl/workers/actor/dp_actor.py` | 门控 PPO + expert BC |
| `quick_start/run_feasibility_pag.sh` | 主训练脚本 |

## Citation (upstream)

```bibtex
@article{jiang2025pag,
  title={PAG: Multi-Turn Reinforced LLM Self-Correction with Policy as Generative Verifier},
  author={Jiang, Yuhua and Xiong, Yuwen and Yuan, Yufeng and Xin, Chao and Xu, Wenyuan and Yue, Yu and Zhao, Qianchuan and Yan, Lin},
  journal={arXiv preprint arXiv:2506.10406},
  year={2025}
}
```
