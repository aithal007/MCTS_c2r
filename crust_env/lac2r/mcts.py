"""
LAC2R MCTS (paper Algorithm 1–2, simplified rollout): UCT selection, root GEN expansion,
child FIX expansion with validator feedback, backpropagation of R from Eq. (5).
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import List, Literal, Optional

from ..verifier import CRustVerifier
from .llm_refiner import LLMRefiner, RefineSession
from .program_context import VerifyOutcome, verify_rust_file
from .safety_reward import (
    baseline_total_from_code,
    compile_score_C,
    node_reward_R,
    safety_ratio_S,
)

NodeKind = Literal["INIT", "GEN", "FIX", "SUCCESS"]


@dataclass
class MCTSNode:
    kind: NodeKind
    code: str
    parent: Optional["MCTSNode"] = None
    children: List["MCTSNode"] = field(default_factory=list)
    n_visits: int = 0
    q_sum: float = 0.0
    s_val: float = 0.0
    c_val: float = 0.0
    session: Optional[RefineSession] = None
    meta: dict = field(default_factory=dict)

    def uct(self, c: float) -> float:
        p = self.parent
        if self.n_visits == 0:
            return float("inf")
        if not p or p.n_visits == 0:
            return self.q()
        return self.q() + c * math.sqrt(
            math.log(p.n_visits + 1) / (self.n_visits + 1e-8)
        )

    def q(self) -> float:
        return 0.0 if self.n_visits == 0 else self.q_sum / self.n_visits

    @property
    def depth(self) -> int:
        d = 0
        p = self.parent
        while p is not None:
            d += 1
            p = p.parent
        return d


@dataclass
class MCTSHyper:
    n_rollouts: int = 10
    uct_c: float = 1.5
    max_depth: int = 5
    w_reward: float = 2.0
    root_branching: int = 4


def _is_terminal(n: MCTSNode) -> bool:
    return n.kind == "SUCCESS"


def _select(root: MCTSNode, c: float) -> List[MCTSNode]:
    path = [root]
    cur = root
    while cur.children and not _is_terminal(cur) and cur.depth < 20:
        cur = max(cur.children, key=lambda ch: ch.uct(c))
        path.append(cur)
    return path


def _backprop(path: List[MCTSNode], reward: float) -> None:
    for n in path:
        n.n_visits += 1
        n.q_sum += reward


def _snapshot_metrics(
    code: str,
    *,
    baseline_total: int,
    compilable: bool,
):
    s_snap = safety_ratio_S(code, baseline_total, compilable=compilable)
    return s_snap.S, s_snap


def run_mcts(
    file_path: str,
    baseline_r0: str,
    verifier: CRustVerifier,
    llm: LLMRefiner,
    require_tests: bool,
    hyp: MCTSHyper,
) -> MCTSNode:
    baseline_total = max(1, baseline_total_from_code(baseline_r0))
    b_out = verify_rust_file(verifier, file_path, baseline_r0, require_tests=require_tests)
    s0, _ = _snapshot_metrics(
        baseline_r0, baseline_total=baseline_total, compilable=b_out.compilable
    )
    c0 = compile_score_C(b_out.n_compile_errors)

    root = MCTSNode(kind="INIT", code=baseline_r0, meta={"s0": s0, "c0": c0, "b_total": baseline_total})
    best_success: Optional[MCTSNode] = None
    best_s: float = -1.0

    for _ in range(hyp.n_rollouts):
        path = _select(root, hyp.uct_c)
        leaf = path[-1]

        if _is_terminal(leaf) or leaf.depth >= hyp.max_depth:
            _backprop(path, 0.0)
            continue

        if not leaf.children:
            if leaf.kind == "INIT":
                for v in range(hyp.root_branching):
                    code = llm.initial_gen(baseline_r0, v)
                    o = verify_rust_file(verifier, file_path, code, require_tests=require_tests)
                    comp = o.compilable
                    S, _s = _snapshot_metrics(code, baseline_total=baseline_total, compilable=comp)
                    c1 = compile_score_C(o.n_compile_errors)
                    r = node_reward_R(c0, c1, s0, S, w=hyp.w_reward)
                    sess = llm.session_after_initial(baseline_r0, code)
                    kind: NodeKind = "SUCCESS" if o.success else ("GEN" if comp else "FIX")
                    child = MCTSNode(
                        kind=kind, code=code, parent=root, s_val=S, c_val=c1, session=sess, meta={}
                    )
                    root.children.append(child)
                    if kind == "SUCCESS" and (best_success is None or S > best_s):
                        best_success = child
                        best_s = S
                    _backprop([root, child], r)
            else:
                if leaf.session is None:
                    _backprop(path, 0.0)
                    continue
                o_leaf = verify_rust_file(verifier, file_path, leaf.code, require_tests=require_tests)
                fb, kind_fb = _feedback_str(o_leaf)
                s_prev, c_prev = leaf.s_val, leaf.c_val
                if not fb:
                    fb = "Refine for fewer raw-pointer / unsafe constructs; preserve public API and tests."
                sess = _copy_session(leaf)
                new_code = llm.fix_with_feedback(sess, feedback=fb, err_kind=kind_fb)
                o2 = verify_rust_file(verifier, file_path, new_code, require_tests=require_tests)
                comp = o2.compilable
                S, _s = _snapshot_metrics(new_code, baseline_total=baseline_total, compilable=comp)
                c1 = compile_score_C(o2.n_compile_errors)
                r = node_reward_R(c_prev, c1, s_prev, S, w=hyp.w_reward)
                nkind: NodeKind = "SUCCESS" if o2.success else ("FIX" if (not comp or not o2.success) else "GEN")
                child = MCTSNode(
                    kind=nkind,
                    code=new_code,
                    parent=leaf,
                    s_val=S,
                    c_val=c1,
                    session=sess,
                    meta={},
                )
                leaf.children.append(child)
                if nkind == "SUCCESS" and (best_success is None or S > best_s):
                    best_success = child
                    best_s = S
                _backprop(path + [child], r)
        else:
            ch = random.choice(leaf.children)
            _backprop(path + [ch], 0.05 * (ch.q() + 0.01))

    root.meta["best_success"] = best_success
    root.meta["best_s"] = best_s
    return root


def _copy_session(leaf: MCTSNode) -> RefineSession:
    from copy import deepcopy

    if leaf.session is None:
        return RefineSession()
    s = deepcopy(leaf.session)
    return s


def _feedback_str(o: VerifyOutcome) -> tuple[str, str]:
    if not o.compile_ok:
        parts = [str((d or {}).get("message", "")) for d in o.diagnostics[:10]]
        return "\n".join(p for p in parts if p), "compiler"
    if o.compile_ok and not o.success and o.test_output:
        return o.test_output[:2000], "test"
    return (
        "Reduce unsafe blocks and raw pointer usage; keep behavior identical to C.",
        "safety",
    )
