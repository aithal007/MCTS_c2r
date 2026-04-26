"""
Paper Equations (3–5):
  S(r_i) = m(r_i) * max(1 - T_i/T_0, 0)   where T = RPC+RPR+LUC+UCE+UTC
  C(r_i) = 1 / (|EC(r_i)| + 1)
  R = C(r_i) - C(r_{i-1}) + w * (S(r_i) - S(r_{i-1}))
"""

from __future__ import annotations

from dataclasses import dataclass

from .unsafe_constructs import UnsafeConstructCounts, count_unsafe_constructs


@dataclass
class SafetySnapshot:
    """Holds S and components for one program state (single target file view)."""

    S: float
    m: int
    counts: UnsafeConstructCounts
    total_unsafe: int

    def as_dict(self) -> dict:
        return {
            "S": self.S,
            "m": self.m,
            "total_unsafe": self.total_unsafe,
            **self.counts.as_dict(),
        }


def _total_T(counts: UnsafeConstructCounts) -> int:
    return counts.total()


def safety_ratio_S(
    rust_code: str,
    baseline_total: int,
    *,
    compilable: bool,
) -> SafetySnapshot:
    """
    Equation (3). Baseline is sum T_0 from initial C2Rust-style translation (r_0).
    If baseline_total == 0, use 1 to avoid division by zero (degenerate safe baseline).
    """
    counts = count_unsafe_constructs(rust_code)
    t_i = _total_T(counts)
    b = max(1, int(baseline_total))
    m = 1 if compilable else 0
    ratio = max(0.0, 1.0 - (t_i / float(b)))
    s = m * ratio
    return SafetySnapshot(S=float(s), m=m, counts=counts, total_unsafe=t_i)


def compile_score_C(n_compile_errors: int) -> float:
    """Equation (4)."""
    n = max(0, int(n_compile_errors))
    return 1.0 / (n + 1)


def node_reward_R(
    c_prev: float,
    c_cur: float,
    s_prev: float,
    s_cur: float,
    *,
    w: float = 2.0,
) -> float:
    """Equation (5)."""
    return (c_cur - c_prev) + w * (s_cur - s_prev)


def baseline_total_from_code(r0_code: str) -> int:
    c = count_unsafe_constructs(r0_code)
    return c.total()
