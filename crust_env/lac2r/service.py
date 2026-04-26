"""
Top-level LAC2R run: MCTS + verifier + best SUCCESS node (paper Algorithm 1, Find_Best_Solution).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List

from ..verifier import CRustVerifier
from .llm_refiner import LAC2RLLMConfig, LLMRefiner
from .mcts import MCTSHyper, MCTSNode, run_mcts
from .program_context import read_file


@dataclass
class LAC2RConfig:
    workspace: str
    file_path: str
    n_rollouts: int = 10
    uct_c: float = 1.5
    max_depth: int = 5
    w_reward: float = 2.0
    root_branching: int = 4
    require_tests: bool = True
    write_best: bool = False


def _collect_successes(root: MCTSNode) -> List[MCTSNode]:
    out: list[MCTSNode] = []
    stack = [root]
    while stack:
        n = stack.pop()
        if n.kind == "SUCCESS":
            out.append(n)
        stack.extend(n.children)
    return out


def find_best_solution(root: MCTSNode, baseline_code: str) -> Dict[str, Any]:
    """
    All SUCCESS test-passing nodes; return arg max S(·) on the snippet, else baseline (paper).
    """
    best_meta = root.meta.get("best_success")
    if isinstance(best_meta, MCTSNode) and best_meta.code:
        return {
            "selected": "mcts",
            "code": best_meta.code,
            "S": best_meta.s_val,
            "node_kind": best_meta.kind,
        }
    nodes = _collect_successes(root)
    if not nodes:
        return {"selected": "baseline", "code": baseline_code, "S": 0.0, "node_kind": "INIT"}
    best = max(nodes, key=lambda x: x.s_val)
    return {"selected": "mcts", "code": best.code, "S": best.s_val, "node_kind": best.kind}


def run_lac2r_refine(cfg: LAC2RConfig) -> Dict[str, Any]:
    workspace = os.path.normpath(cfg.workspace)
    r0 = read_file(workspace, cfg.file_path)
    verifier = CRustVerifier(workspace)
    llm = LLMRefiner(LAC2RLLMConfig.from_env())
    hyp = MCTSHyper(
        n_rollouts=cfg.n_rollouts,
        uct_c=cfg.uct_c,
        max_depth=cfg.max_depth,
        w_reward=cfg.w_reward,
        root_branching=cfg.root_branching,
    )
    root = run_mcts(
        cfg.file_path,
        r0,
        verifier,
        llm,
        cfg.require_tests,
        hyp,
    )
    best = find_best_solution(root, r0)
    if cfg.write_best and str(best.get("selected")) == "mcts" and best.get("code") is not None:
        p = os.path.join(workspace, cfg.file_path)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            f.write(str(best["code"]))

    return {
        "ok": True,
        "file_path": cfg.file_path,
        "result": best,
        "mcts": {
            "best_S_tracked": root.meta.get("best_s"),
            "openai_configured": bool(os.getenv("OPENAI_API_KEY")),
        },
    }
