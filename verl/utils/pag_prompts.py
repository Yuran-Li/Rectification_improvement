"""Canonical PAG user templates. Keep RL rollout in sync.

Verify is one generation: a short error critique (feedback) followed by a
hard closer. Do not emit a full corrected solution or \\boxed{} in verify.
"""

VERIFY_USER = (
    "Verify the previous solution without re-solving the problem from scratch. "
    "Check the given solution step-by-step: if you find a mistake, state the wrong step, "
    "If you find a mistake: in 1-4 sentences, name the wrong step, explain why it is wrong, "
    "explain why it is wrong, and end your response with 'The answer is wrong'. "
    "If all steps are correct: do not propose edits. End your response with: The answer is correct."
)

REGENERATE_USER = (
    "You indicated that your previous answer was wrong. "
    "Please provide the correct solution to the math problem."
)

# Fixed counterfactual critique (not sampled). Used only for R_critique:
# R_critique = R_y(y_self) - R_y(y_generic), placed on the self-feedback span.
GENERIC_CRITIQUE = (
    "The previous solution contains a mistake in its reasoning. The incorrect step should be fixed.\n"
    "The answer is wrong."
)