"""
Connect LAC2R paper safety S(r) to the existing CRust RL reward (shaping only).

Set env var `CRUST_LAC2R_REWARD=1` to add a small bonus proportional to
Equation (3) safety ratio for the current Rust file vs the baseline captured at reset.
This keeps GRPO / env.step() as the training loop while steering policy toward
fewer unsafe constructs.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional, Tuple

from .safety_reward import baseline_total_from_code, safety_ratio_S


W_LAC2R = float(os.getenv("CRUST_LAC2R_WEIGHT", "0.06"))


def lac2r_shaping_add(
    code: str,
    baseline_r0: str,
    verification: Dict[str, Any],
) -> Tuple[float, Dict[str, float]]:
    if os.getenv("CRUST_LAC2R_REWARD", "0") not in ("1", "true", "yes"):
        return 0.0, {}
    stage = str(verification.get("stage", ""))
    comp = stage in ("testing", "complete")
    bt = max(1, baseline_total_from_code(baseline_r0 or code))
    S = safety_ratio_S(code, bt, compilable=comp).S
    bonus = W_LAC2R * float(S)
    return bonus, {"lac2r_s_shaping": round(bonus, 4), "lac2r_S": round(float(S), 4)}
