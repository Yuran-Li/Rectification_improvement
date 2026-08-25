#!/usr/bin/env python3
"""Real branching dump for Step-4 (chat-message based; full-concat).

Uses merged dual-head critic (FSDP step100) + HF actor (Math-7B or RL actor).

  CUDA_VISIBLE_DEVICES=5,6 PYTHONUNBUFFERED=1 PYTHONPATH=. \\
    python tools/run_vf_branching_dump.py --n-states 60 --k 4 --out results/vf_branching_step100.jsonl
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForTokenClassification, AutoTokenizer

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from verl.utils.reward_score.math_verify import compute_score as get_policy_score

VERIFY_TMPL = (
    "Verify the previous solution without re-solving the problem from scratch. "
    "Check the given solution step-by-step: if you find a mistake, state the wrong step, "
    "explain why it is wrong, and end your response with 'The answer is wrong'. "
    "If all steps are correct, end your response with 'The answer is correct'."
)
REGEN_TMPL = (
    "You indicated that your previous answer was wrong. "
    "Please provide the correct solution to the math problem."
)


def log(msg: str) -> None:
    print(msg, flush=True)


def msgs_from_prompt(prompt_col) -> List[Dict[str, str]]:
    return [dict(m) for m in list(prompt_col)]


def render_ids(tokenizer, messages: List[Dict[str, str]], add_generation_prompt: bool) -> List[int]:
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=add_generation_prompt
    )
    return tokenizer.encode(text, add_special_tokens=False)


def eos_ids(tokenizer) -> List[int]:
    ids: List[int] = []
    for t in (getattr(tokenizer, "eos_token", None), "<|im_end|>", "<|endoftext|>"):
        if not t:
            continue
        tid = tokenizer.convert_tokens_to_ids(t)
        if isinstance(tid, int) and tid >= 0 and tid != getattr(tokenizer, "unk_token_id", -1):
            ids.append(tid)
    if tokenizer.eos_token_id is not None:
        ids.append(int(tokenizer.eos_token_id))
    out: List[int] = []
    for i in ids:
        if i not in out:
            out.append(i)
    return out or [tokenizer.pad_token_id]


@torch.no_grad()
def generate_assistant(
    model,
    tokenizer,
    messages: List[Dict[str, str]],
    *,
    max_new: int,
    temperature: float,
    top_p: float,
) -> str:
    device = next(model.parameters()).device
    prefix = render_ids(tokenizer, messages, add_generation_prompt=True)
    inp = torch.tensor([prefix], device=device, dtype=torch.long)
    out = model.generate(
        input_ids=inp,
        attention_mask=torch.ones_like(inp),
        max_new_tokens=max_new,
        do_sample=True,
        temperature=max(temperature, 1e-5),
        top_p=top_p,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=eos_ids(tokenizer),
    )
    gen = out[0, len(prefix) :].tolist()
    stop = set(eos_ids(tokenizer) + [tokenizer.pad_token_id])
    while gen and gen[-1] in stop:
        gen.pop()
    return tokenizer.decode(gen, skip_special_tokens=True).strip()


def judgment_from_text(text: str) -> Optional[str]:
    matches = list(re.finditer(r"The answer is (correct|wrong)\.", text))
    if not matches:
        return None
    return matches[-1].group(1).lower()


def answer_fail(text: str, gt: str) -> int:
    try:
        acc = float(get_policy_score(solution_str=text, ground_truth=gt)["acc"])
    except Exception:
        acc = 0.0
    return 1 if acc < 0.5 else 0


@torch.no_grad()
def score_vf_at_end(critic, tokenizer, messages: List[Dict[str, str]], device) -> Tuple[float, List[int]]:
    """Score V_F on last token of the rendered conversation (no generation prompt)."""
    ids = render_ids(tokenizer, messages, add_generation_prompt=False)
    t = torch.tensor([ids], device=device, dtype=torch.long)
    logits = critic(input_ids=t, attention_mask=torch.ones_like(t)).logits
    vf = float(torch.sigmoid(logits[0, -1, 1]).item())
    return vf, ids


def ensure_merged_critic(ckpt_dir: Path, merged_path: Path, world_size: int = 8) -> Path:
    if merged_path.is_file():
        log(f"using existing merged critic: {merged_path}")
        return merged_path
    from tools.merge_fsdp_shards import merge_shards

    log(f"merging FSDP critic from {ckpt_dir}")
    merged = merge_shards(ckpt_dir, world_size=world_size)
    merged_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(merged, merged_path)
    return merged_path


def seed_and_states(
    actor,
    tokenizer,
    base_msgs: List[Dict[str, str]],
    gt: str,
    *,
    num_turns: int,
    max_new: int,
    temperature: float,
    top_p: float,
    force_rectify: bool,
) -> Dict[str, Any]:
    """Return decision states as message snapshots."""
    msgs = deepcopy(base_msgs)
    states: List[Dict[str, Any]] = []

    y0 = generate_assistant(actor, tokenizer, msgs, max_new=max_new, temperature=temperature, top_p=top_p)
    msgs.append({"role": "assistant", "content": y0})
    states.append({"role": "V", "messages": deepcopy(msgs), "needs": "verify", "answer": y0})

    did_force = False
    last_y = y0
    for turn in range(num_turns):
        msgs.append({"role": "user", "content": VERIFY_TMPL})
        v = generate_assistant(
            actor, tokenizer, msgs, max_new=min(256, max_new), temperature=temperature, top_p=top_p
        )
        msgs.append({"role": "assistant", "content": v})
        j = judgment_from_text(v)
        should = (force_rectify and not did_force) or (j == "wrong")
        if (not should) or turn >= num_turns - 1:
            break
        states.append({"role": "R", "messages": deepcopy(msgs), "needs": "rectify", "answer": last_y})
        msgs.append({"role": "user", "content": REGEN_TMPL})
        y = generate_assistant(actor, tokenizer, msgs, max_new=max_new, temperature=temperature, top_p=top_p)
        msgs.append({"role": "assistant", "content": y})
        last_y = y
        states.append({"role": "V", "messages": deepcopy(msgs), "needs": "verify", "answer": last_y})
        did_force = True

    return {"states": states, "seed_fail": answer_fail(last_y, gt), "last_y": last_y}


def branch_continue(
    actor,
    tokenizer,
    state: Dict[str, Any],
    gt: str,
    *,
    num_turns_left: int,
    max_new: int,
    temperature: float,
    top_p: float,
) -> int:
    msgs = deepcopy(state["messages"])
    last_y = state.get("answer", "")

    if state["needs"] == "verify":
        msgs.append({"role": "user", "content": VERIFY_TMPL})
        v = generate_assistant(
            actor, tokenizer, msgs, max_new=min(256, max_new), temperature=temperature, top_p=top_p
        )
        msgs.append({"role": "assistant", "content": v})
        j = judgment_from_text(v)
        if j == "wrong" and num_turns_left > 0:
            msgs.append({"role": "user", "content": REGEN_TMPL})
            last_y = generate_assistant(
                actor, tokenizer, msgs, max_new=max_new, temperature=temperature, top_p=top_p
            )
            msgs.append({"role": "assistant", "content": last_y})
            for _ in range(num_turns_left - 1):
                msgs.append({"role": "user", "content": VERIFY_TMPL})
                v = generate_assistant(
                    actor, tokenizer, msgs, max_new=min(256, max_new), temperature=temperature, top_p=top_p
                )
                msgs.append({"role": "assistant", "content": v})
                if judgment_from_text(v) != "wrong":
                    break
                msgs.append({"role": "user", "content": REGEN_TMPL})
                last_y = generate_assistant(
                    actor, tokenizer, msgs, max_new=max_new, temperature=temperature, top_p=top_p
                )
                msgs.append({"role": "assistant", "content": last_y})
        # if verify says correct, final answer stays last_y
    else:  # rectify
        msgs.append({"role": "user", "content": REGEN_TMPL})
        last_y = generate_assistant(actor, tokenizer, msgs, max_new=max_new, temperature=temperature, top_p=top_p)
        msgs.append({"role": "assistant", "content": last_y})
        for _ in range(max(num_turns_left - 1, 0)):
            msgs.append({"role": "user", "content": VERIFY_TMPL})
            v = generate_assistant(
                actor, tokenizer, msgs, max_new=min(256, max_new), temperature=temperature, top_p=top_p
            )
            msgs.append({"role": "assistant", "content": v})
            if judgment_from_text(v) != "wrong":
                break
            msgs.append({"role": "user", "content": REGEN_TMPL})
            last_y = generate_assistant(
                actor, tokenizer, msgs, max_new=max_new, temperature=temperature, top_p=top_p
            )
            msgs.append({"role": "assistant", "content": last_y})

    return answer_fail(last_y, gt)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--actor", default="/data/yuranli/LLM/2026.04/models/Qwen2.5-Math-7B-Instruct")
    ap.add_argument(
        "--critic-fsdp",
        default="checkpoints/Rectification_Feasibility/qwen25math7b_feas_pag_t4/global_step_100/critic",
    )
    ap.add_argument("--critic-merged", default="results/merged_critic_feas_step100.pt")
    ap.add_argument("--data", default="datasets/math500.parquet")
    ap.add_argument("--out", default="results/vf_branching_step100.jsonl")
    ap.add_argument("--n-states", type=int, default=60)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--max-problems", type=int, default=24)
    ap.add_argument("--num-turns", type=int, default=2)
    ap.add_argument("--max-new", type=int, default=384)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--actor-device", default="cuda:0")
    ap.add_argument("--critic-device", default="cuda:1")
    ap.add_argument("--no-force-rectify", action="store_true")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    critic_pt = ensure_merged_critic(Path(args.critic_fsdp), Path(args.critic_merged))

    log("loading tokenizer/actor...")
    tok = AutoTokenizer.from_pretrained(args.actor, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    actor = AutoModelForCausalLM.from_pretrained(
        args.actor, torch_dtype=torch.bfloat16, trust_remote_code=True
    ).to(args.actor_device)
    actor.eval()

    log("loading critic (this can take a few minutes for 28GB pt)...")
    cfg = AutoConfig.from_pretrained(args.actor, trust_remote_code=True)
    cfg.num_labels = 2
    cfg.classifier_dropout = 0.0
    critic = AutoModelForTokenClassification.from_pretrained(
        args.actor, config=cfg, torch_dtype=torch.bfloat16, trust_remote_code=True, ignore_mismatched_sizes=True
    )
    state = torch.load(critic_pt, map_location="cpu", weights_only=False)
    missing, unexpected = critic.load_state_dict(state, strict=False)
    log(f"critic load: missing={len(missing)} unexpected={len(unexpected)}")
    critic.to(args.critic_device)
    critic.eval()

    df = pd.read_parquet(args.data)
    idxs = np.random.permutation(len(df))[: args.max_problems]

    rows: List[Dict[str, Any]] = []
    n_sv = n_sr = 0
    force = not args.no_force_rectify

    for pi, i in enumerate(idxs):
        if len(rows) >= args.n_states:
            break
        row = df.iloc[int(i)]
        gt = row["reward_model"]["ground_truth"]
        uid = str(row.get("unique_id", i))
        log(f"[{pi+1}/{len(idxs)}] seed {uid} collected={len(rows)}")
        try:
            seed = seed_and_states(
                actor,
                tok,
                msgs_from_prompt(row["prompt"]),
                gt,
                num_turns=args.num_turns,
                max_new=args.max_new,
                temperature=args.temperature,
                top_p=args.top_p,
                force_rectify=force,
            )
        except Exception as e:
            log(f"  seed failed: {e}")
            continue

        for si, st in enumerate(seed["states"]):
            if len(rows) >= args.n_states:
                break
            try:
                vf, _ = score_vf_at_end(critic, tok, st["messages"], args.critic_device)
            except Exception as e:
                log(f"  vf failed: {e}")
                continue
            fails = []
            for _k in range(args.k):
                try:
                    fails.append(
                        branch_continue(
                            actor,
                            tok,
                            st,
                            gt,
                            num_turns_left=args.num_turns,
                            max_new=args.max_new,
                            temperature=args.temperature,
                            top_p=args.top_p,
                        )
                    )
                except Exception as e:
                    log(f"  branch err: {e}")
                    fails.append(1)
            rec = {
                "state_id": f"{uid}_{st['role']}_{si}",
                "role": st["role"],
                "vf": vf,
                "branch_fail": fails,
                "meta": {
                    "unique_id": uid,
                    "seed_fail": seed["seed_fail"],
                    "p_hat": float(np.mean(fails)),
                    "answer_preview": (st.get("answer") or "")[:160],
                    "actor": args.actor,
                    "critic": str(critic_pt),
                },
            }
            rows.append(rec)
            if st["role"] == "V":
                n_sv += 1
            else:
                n_sr += 1
            log(
                f"  + {rec['state_id']} vf={vf:.3f} p̂={rec['meta']['p_hat']:.3f} "
                f"fails={fails} nV={n_sv} nR={n_sr}"
            )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    log(f"wrote {out} n={len(rows)} nV={n_sv} nR={n_sr}")


if __name__ == "__main__":
    main()
