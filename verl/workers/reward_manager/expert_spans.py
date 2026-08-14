"""Positive-behavior expert spans for same-UID bootstrap replay.

When F(s_A)>0 and a same-UID sibling τ_B+ is final-correct, replay τ_B+
with role masks that match whatever valid behavior that sibling discovered:

  first-shot:  x → y^C → v^accept
      BC generator y. Verifier accept BC is controlled by bc_verifier_accept:
      when False (recommended), only the solution is replayed — the model
      must learn "accept" via RL reward alone, not by BC from a gated-open
      state that V_F already flagged as high-risk.

  I→C:        W → true reject → C
      BC the true-reject verifier span and the successful rectifier span.
      This path is unaffected by bc_verifier_accept (reject ≠ accept).

Wrong answers / wrong verifies / rectifier after false-reject are never painted.
Only call this on a trajectory that is (or will be checked as) final-correct.
"""
from __future__ import annotations

from typing import Tuple


def positive_expert_roles(
    *,
    y_correct: bool,
    verify_oracle: bool,
    verify_accept: bool,
    has_rectify: bool,
    rectify_correct: bool = False,
    bc_verifier_accept: bool = True,
) -> Tuple[bool, bool, bool]:
    """Which roles to paint for one (answer, verify, optional rectify) round.

    Returns (paint_y, paint_v, paint_r).

    bc_verifier_accept: if False, first-shot correct siblings replay only the
        solution (y), not the verifier accept token span.  I→C true-reject is
        unaffected — that verifier is a reject, not an accept, and its BC is
        always desirable.
    """
    if not has_rectify:
        if y_correct and verify_oracle and verify_accept:
            return True, bc_verifier_accept, False
        return False, False, False
    true_reject = bool(verify_oracle) and (not verify_accept)
    if true_reject and rectify_correct:
        # Do not BC the wrong previous answer as a generator expert.
        return False, True, True
    return False, False, False
