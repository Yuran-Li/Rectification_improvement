#!/usr/bin/env python3
"""Merge world_size=8 FSDP DTensor shards → plain state_dict (CPU).

Example:
  PYTHONPATH=. python tools/merge_fsdp_shards.py \\
    --ckpt checkpoints/Rectification_Feasibility/qwen25math7b_feas_pag_t4/global_step_100/critic \\
    --out results/merged_critic_step100.pt
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.distributed.tensor  # noqa: F401 — needed to unpickle DTensors


def merge_shards(ckpt_dir: Path, world_size: int = 8) -> dict:
    shards = []
    for r in range(world_size):
        p = ckpt_dir / f"model_world_size_{world_size}_rank_{r}.pt"
        print(f"loading {p} ...")
        shards.append(torch.load(p, map_location="cpu", weights_only=False))
    keys = list(shards[0].keys())
    merged = {}
    for i, k in enumerate(keys):
        ref = shards[0][k]
        gshape = tuple(ref.shape)
        locals_ = [s[k].to_local().contiguous() for s in shards]
        nonempty = [t for t in locals_ if t.numel() > 0]
        if not nonempty:
            raise RuntimeError(f"empty shards for {k}")
        if len(nonempty) == 1:
            tens = nonempty[0]
        else:
            # FSDP shard dim from placement
            dim = 0
            try:
                dim = int(ref.placements[0].dim)
            except Exception:
                dim = 0
            tens = torch.cat(nonempty, dim=dim)
        if tuple(tens.shape) != gshape:
            # trim padding if any
            slices = [slice(0, gshape[d]) for d in range(len(gshape))]
            tens = tens[tuple(slices)]
        if tuple(tens.shape) != gshape:
            raise RuntimeError(f"{k}: merged {tuple(tens.shape)} != global {gshape}")
        merged[k] = tens.cpu()
        if (i + 1) % 50 == 0:
            print(f"  merged {i+1}/{len(keys)}")
    print(f"done: {len(merged)} tensors")
    return merged


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--world-size", type=int, default=8)
    args = ap.parse_args()
    merged = merge_shards(Path(args.ckpt), world_size=args.world_size)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(merged, out)
    print(f"wrote {out} ({out.stat().st_size/1e9:.2f} GB)")


if __name__ == "__main__":
    main()
