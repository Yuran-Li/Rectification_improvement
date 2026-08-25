#!/usr/bin/env python3
"""Step-4 branching calibration CLI.

Modes
-----
1) Synthetic dry-run (no GPU) — verifies metrics + pass criteria::

    PYTHONPATH=. python tools/vf_branching_audit.py --synthetic --n 80 --k 8

2) Offline JSONL — each line::

    {"state_id": "...", "role": "V"|"R", "vf": 0.72, "branch_fail": [1,0,1,1]}

    branch_fail[k]=1 means continuation k ended with wrong final answer (G_F=1).

    PYTHONPATH=. python tools/vf_branching_audit.py --jsonl path/to/states.jsonl --eps 0.3

3) Online branching (optional, needs vLLM + critic) — not wired by default.
   Dump states from a CRITIC_ONLY run, then branch offline / with your own generator;
   feed results back as JSONL into mode 2.

Exit code 0 iff pass_checks all True (use --no-fail-exit to always 0).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import numpy as np

from verl.trainer.ppo.vf_calibration import (
    audit_branching_calibration,
    format_report,
    simulate_branch_states,
    states_from_jsonl_rows,
)


def _load_jsonl(path: Path):
    if not path.is_file():
        raise FileNotFoundError(
            f"JSONL not found: {path}\n"
            "  --jsonl expects a real dump file you produced after branching.\n"
            "  Quick checks (no dump needed):\n"
            "    PYTHONPATH=. python tools/vf_branching_audit.py --synthetic --n 80 --k 8\n"
            "  Example file format:\n"
            "    examples/vf_branching_states.example.jsonl\n"
            "  Each line: "
            '{"state_id":"...","role":"V"|"R","vf":0.72,"branch_fail":[1,0,1,1]}'
        )
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    if not rows:
        raise ValueError(f"JSONL is empty: {path}")
    return rows


def main():
    ap = argparse.ArgumentParser(description="V_F branching calibration audit (Step-4)")
    ap.add_argument("--synthetic", action="store_true", help="calibrated synthetic demo")
    ap.add_argument("--miscalibrated", action="store_true", help="constant-VF demo (expect FAIL)")
    ap.add_argument("--jsonl", type=str, default=None, help="offline branch dump")
    ap.add_argument("--n", type=int, default=80, help="synthetic #states")
    ap.add_argument("--k", type=int, default=8, help="branches per state")
    ap.add_argument("--eps", type=float, default=0.3, help="gate threshold ε")
    ap.add_argument("--ece-max", type=float, default=0.15)
    ap.add_argument("--corr-min", type=float, default=0.3)
    ap.add_argument("--gate-gap-min", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default=None, help="write report JSON")
    ap.add_argument("--no-fail-exit", action="store_true")
    args = ap.parse_args()

    if args.jsonl:
        rows = _load_jsonl(Path(args.jsonl))
        states = states_from_jsonl_rows(rows)
    elif args.synthetic or args.miscalibrated:
        true_p = np.linspace(0.05, 0.95, args.n)
        if args.miscalibrated:
            vf = np.full(args.n, 0.5)
            states = simulate_branch_states(true_p, vf=vf, k=args.k, seed=args.seed)
        else:
            states = simulate_branch_states(true_p, vf=true_p, k=args.k, seed=args.seed)
    else:
        ap.error("pass --synthetic, --miscalibrated, or --jsonl PATH")

    report = audit_branching_calibration(
        states,
        eps=args.eps,
        ece_max=args.ece_max,
        corr_min=args.corr_min,
        gate_gap_min=args.gate_gap_min,
    )
    print(format_report(report))
    if args.out:
        Path(args.out).write_text(json.dumps(report.to_dict(), indent=2))
        print(f"wrote {args.out}")

    ok = all(report.pass_checks.values())
    if not ok and not args.no_fail_exit:
        sys.exit(1)


if __name__ == "__main__":
    main()
