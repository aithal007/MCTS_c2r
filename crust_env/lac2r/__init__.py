"""
LAC2R — search-based multi-trajectory refinement for safer C-to-Rust translation
(MCTS-guided LLM refinement), aligned with Sim et al., arXiv:2505.15858.

This package layers on top of the existing CRust verifier (cargo check / cargo test)
and static metrics, without replacing the OpenEnv / GRPO training path.
"""

from .safety_reward import (
    UnsafeConstructCounts,
    count_unsafe_constructs,
    safety_ratio_S,
    compile_score_C,
    node_reward_R,
)
from .mcts import LAC2RMCTSEngine, MCTSHyper, MCTSNode, run_mcts
from .service import LAC2RConfig, find_best_solution, run_lac2r_refine
from .rl_bridge import lac2r_shaping_add

__all__ = [
    "UnsafeConstructCounts",
    "count_unsafe_constructs",
    "safety_ratio_S",
    "compile_score_C",
    "node_reward_R",
    "LAC2RConfig",
    "find_best_solution",
    "run_lac2r_refine",
    "LAC2RMCTSEngine",
    "MCTSHyper",
    "MCTSNode",
    "run_mcts",
    "lac2r_shaping_add",
]
