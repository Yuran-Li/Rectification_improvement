# Feasibility-Guided Self-Correction

多角色联合自纠错：generate / verify / rectify 共用一个策略。用 **role-aware, state-level feasibility** 决定何时做 recovery supervision（不硬关 PPO）。

**正式目标（不是 global CMDP）：** 用 \(V_F(s)\) 找出当前 policy **不可靠的自主 recovery region**，在那些 decision state 上加强 recovery 监督。

\[
F^\pi(s)=V_F^\pi(s)-\varepsilon,\qquad
g(s)=\mathbf{1}[V_F^\pi(s)>\varepsilon]
\]

\[
L_{\mathrm{actor}}=L_{\mathrm{PPO}}+\beta\,g(s)\,L_{\mathrm{recovery}}
\]

默认 **无** global Lagrangian / \(J_F\le B_F\)（可用 `USE_LAGRANGIAN=True` 做 ablation）。

| 模块 | 做法 |
|------|------|
| 轨迹 | **全量拼接**（默认 `SLIDE_WINDOW=False`）：整条 traj 拼成一条；角色决策点仍是 \(s^V\)@答案末、\(s^R\)@verify 末 |
| Critic | 双头 \(V_R,V_F\)。\(V_F^\pi(s)=P_\pi(z_{\mathrm{final}}=0\mid s)\)，\(G_F(s)=\mathbf{1}[z_{\mathrm{final}}=0]\)（最终答案错；同 traj 上 \(s^V/s^R\) 共享同一 target） |
| \(G_F\) | \(z_{\mathrm{final}}=1\) iff traj **最终答案**正确。不是“曾经失败 / role error”。见 `tests/test_vf_targets_final_answer.py` |
| 观测 \(s\) | **全量拼接**（`slide_window=False`）：\(s\) 是到 role boundary 为止的 causal prefix（含更早轮）。角色仍是 \(s^V\)@答案末 / \(s^R\)@verify 末。见 `tests/test_vf_state_prefix_dump.py` |
| Reward | utility-aware：rectify \(R=\hat R+\alpha\Delta\)，verify \(R=\hat R_v+\beta U\) |
| 门控 | \(F(s)\le0\) → 该决策 **仅 PPO**；\(F(s)>\varepsilon\) → **PPO +** 该角色的 recovery 项（见 Expert） |
| Expert | **两级，勿混为一谈**：① self-bootstrap = *problem-conditioned* positive replay（\(s_B\neq s_A\)）；② GPT = *state-conditioned* \(a_E\sim\pi_E(\cdot\mid s_A)\)（默认关） |

### \(V_F\) target（务必写清）

对任意 decision state \(s\)：

\[
G_F(s)=\mathbf{1}[\text{policy fails to reach/retain a correct answer in the remaining correction horizon}].
\]

- **Verification state** \(s_i^V=(x,y_i)\)：看后续 \(v_i,y_{i+1},v_{i+1},\ldots,y_T\) 最终自主纠错是否成功。
- **Rectification state** \(s_i^R=(x,y_i,v_i)\)：看后续 \(y_{i+1},v_{i+1},\ldots,y_T\) 最终是否成功。

实现上 \(V_F\) 头输出 logits，用 **BCE-with-logits** 拟合 \(G_F\in\{0,1\}\)；门控使用 \(\sigma(V_F^{\mathrm{logit}})\in[0,1]\) 与阈值 \(\varepsilon\) 比较。

### Expert：bootstrap ≠ GPT

| 路径 | 条件 | 学到的是 |
|------|------|----------|
| Self-bootstrap | 同题 \(n\) 条里有 **final-correct** sibling \(\tau_B^+\) | \(s_B\to a_B^+\) 的 **problem-conditioned** replay；**不是**严格的 \(a_E\mid s_A\)。按类型打 mask：first-shot \(y^C\to v^{\mathrm{accept}}\) 只 BC generator + 真接受（**不造 rectifier**）；I→C 才 BC true-reject verify + 成功 rectifier |
| GPT | \(F(s_A)>\varepsilon\) 且缺 bootstrap expert | \(v^E\sim\pi_E(\cdot\mid x,y_i)\) 或 \(y^E\sim\pi_E(\cdot\mid x,y_i,v_i^A)\)，**state-conditioned** |

门控句「\(F(s)>\varepsilon\Rightarrow\) 该角色 recovery BC」对 **GPT** 严格成立；对 bootstrap 只表示「该样本被标成需要 recovery 信号时，可用同题成功轨迹做便宜的正例 replay」。

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

默认：`dual_critic`、`expert_bc`、`bootstrap_same_uid`、`USE_LAGRANGIAN=False`、`EXPERT_BC_LIGHT=0`（**无 API**）。  
`num_turns` 见脚本；**默认 `SLIDE_WINDOW=False`（整条 traj 全量拼接）**。不再依赖 slide_window 的 Markov 截断。

```bash
# 消融（不调 GPT）— 仅 PPO + self-bootstrap
ONLINE_GPT_EXPERT=False bash quick_start/run_feasibility_pag.sh

# 需要 state-conditioned expert 时再开 GPT
export OPENAI_API_KEY=...
ONLINE_GPT_EXPERT=True ONLINE_GPT_MAX_PER_STEP=8 \
  bash quick_start/run_feasibility_pag.sh
```

看日志：`feasibility/bootstrap_transfer_filled`、`actor/expert_bc_loss`、`feasibility/online_gpt_filled_{v,r}`。  
常用变量：`NUM_TURNS`、`N_SAMPLES`（建议 ≥4）、**`FEAS_THRESHOLD` / `ε`**（旧名 `COST_BUDGET` 仍可用）、`EXPERT_BC_COEF`、`EXPERT_BC_LIGHT`（默认 0）。

### 3. 测试（按关卡）

```bash
# Step-1: G_F target table
PYTHONPATH=. python tests/test_vf_targets_final_answer.py
# Step-2: full-concat prefix + VF mask boundaries
PYTHONPATH=. python tests/test_vf_state_prefix_dump.py
# Step-3: isolated V_F BCE (actor frozen; loss↓ / fail>succ / no collapse / no dilution)
PYTHONPATH=. python tests/test_vf_isolated_bce_train.py
# Step-4: branching calibration V_F(s) vs empirical p̂_fail(s)
PYTHONPATH=. python tests/test_vf_branching_calibration.py
PYTHONPATH=. python tools/vf_branching_audit.py --synthetic --n 80 --k 8 --eps 0.3
# Step-5: gate sanity g=1[V_F>ε]; P(fail|g=1)>P(fail|g=0); EXPERT_BC_LIGHT=0
PYTHONPATH=. python tests/test_gate_sanity_metrics.py
# Step-6: self-bootstrap coverage P(∃ sibling success | V_F>ε) by s^V/s^R (GPT off)
PYTHONPATH=. python tests/test_bootstrap_coverage.py
# Step-7: role BC mask (gated s^V→verify tokens only; gated s^R→rectify only);
#         one-step logπ(a_E|s)↑; denom = BC mask mass not B·L
PYTHONPATH=. python tests/test_bc_role_routing.py
# 其它 pack / gate 工具测
PYTHONPATH=. python tests/test_slide_window_pack.py
```

训练日志每 step 打印 `[gate] ...`（在 constraint warmup 强制 feasible **之前**）：
`gate_verify_rate` / `gate_rectify_rate`、gated/ungated 的 mean \(V_F\)、
`P(fail|g=1)` / `P(fail|g=0)`、`sanity_Pfail_g1_gt_g0`。
论文门控 \(g(s)=\mathbf{1}[V_F>\varepsilon]\)；存盘 `feas_gate_*=1[V_F\le\varepsilon]`（取反）。
`F\le0`→PPO only，`F>0`→PPO+BC（默认 `EXPERT_BC_LIGHT=0`）。

Step-6 每 step 打印 `[bootstrap] s_B→a_B+ (NOT s_A→a_E)`：
`cov_v` / `cov_r` = \(P(\exists\) successful sibling \(\mid V_F>\varepsilon)\) 按 \(s^V/s^R\)；
`gpt_needed_*` = 1−coverage（无正例 sibling 的 gated 比例）。GPT 默认关。

实机只训 critic（冻 actor / 关 BC）看 `vf_audit/*`：

```bash
CRITIC_ONLY=1 TOTAL_EPOCHS=1 TRAIN_BS=32 N_GPUS=8 bash quick_start/run_feasibility_pag.sh
```

看：`critic/vf_loss_f` 下降、`vf_audit/E_vf_G_gap>0`、`vf_audit/n_sv`/`n_sr`、`vf_audit/vf_std`。

Step-4 实机：对 50–100 个 \(s^V/s^R\) 各 branch \(K{=}4{-}8\) 条 continuation，dump JSONL（`vf`, `branch_fail`），再：

```bash
# 合并 FSDP critic（一次性）+ 实机 branching dump
PYTHONPATH=. python tools/merge_fsdp_shards.py \
  --ckpt checkpoints/Rectification_Feasibility/qwen25math7b_feas_pag_t4/global_step_100/critic \
  --out results/merged_critic_feas_step100.pt
CUDA_VISIBLE_DEVICES=5,6 PYTHONUNBUFFERED=1 PYTHONPATH=. \
  python tools/run_vf_branching_dump.py --n-states 60 --k 4 \
  --out results/vf_branching_step100.jsonl
PYTHONPATH=. python tools/vf_branching_audit.py --jsonl results/vf_branching_step100.jsonl --eps 0.3
```

通过标准：`corr(V_F,p̂)`↑、`ECE`↓、`P̂(fail|V_F>ε)>P̂(fail|V_F≤ε)`，且 sV/sR 都有样本。  
示例/合成：`examples/vf_branching_states.example.jsonl`、`--synthetic`。

## 关键代码

| 路径 | 作用 |
|------|------|
| `verl/workers/rollout/vllm_rollout/vllm_pag_rollout_spmd.py` | 多轮 rollout（可选 slide_window） |
| `verl/workers/reward_manager/pag.py` | utility reward；\(G_F\) @ \(s^V\)/\(s^R\)；role expert mask |
| `verl/workers/critic/dp_critic.py`、`fsdp_workers.py` | 双头 \(V_R/V_F\) |
| `verl/trainer/ppo/ray_trainer.py` | role-aware \(F(s)\) 门控；可选 Lagrangian ablation；GPT 接入 |
| `verl/trainer/ppo/online_gpt_expert.py` | state-conditioned GPT（verify **或** rectify） |
| `verl/trainer/ppo/expert_buffer.py` | same-uid bootstrap transfer（problem-conditioned） |
| `verl/workers/actor/dp_actor.py` | PPO + role-gated recovery BC |
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
