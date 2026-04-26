"""
LAC2R — Monte Carlo Tree Search for multi-trajectory Rust refinement (Sim et al., arXiv:2505.15858).

Implements the structure of:
  - Algorithm 1: MCTS_Search, Rollout(Select → Expand → Simulate → Backpropagate)
  - Algorithm 2: Expand (Init → heterogeneous GEN children; else LLM + insert program;
    compile / test feedback → Fix or Success; reward from Eq. 3–5)

Simulate: greedy “virtual” rollout from a newly expanded child (optional N steps) that
reuses the same fix/refine LLM path without materializing the full sim subtree; the
simulation return value is backpropagated with the expand edge (paper: path_s + path_r).
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from copy import deepcopy
from typing import List, Literal, Optional, Tuple

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
    # True once Expand() has run for this node (avoids re-root-spam)
    expanded: bool = False

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
        d, p = 0, self.parent
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
    """Greedy virtual simulation steps after each expand (0 = off)."""
    simulation_depth: int = 0
    """Probability to use a no-feedback (null, LLM_k) child at non-root when expanding."""
    p_no_feedback_expand: float = 0.0


def _is_terminal(n: MCTSNode) -> bool:
    return n.kind == "SUCCESS"


def _select_to_leaf(root: MCTSNode, c: float) -> List[MCTSNode]:
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


def _snap(
    code: str,
    *,
    baseline_total: int,
    compilable: bool,
) -> Tuple[float, float]:
    s_snap = safety_ratio_S(code, baseline_total, compilable=compilable)
    return s_snap.S, s_snap


def _feedback_str(o: VerifyOutcome) -> tuple[str, str]:
    if not o.compile_ok:
        parts = [str((d or {}).get("message", "")) for d in o.diagnostics[:10]]
        return "\n".join(p for p in parts if p), "compiler"
    if o.compile_ok and not o.success and o.test_output:
        return o.test_output[:2000], "test"
    return (
        "Reduce raw pointers, unsafe, and type casts; preserve API and tests.",
        "safety",
    )


def _copy_session(leaf: MCTSNode) -> RefineSession:
    if leaf.session is None:
        return RefineSession()
    return deepcopy(leaf.session)


class LAC2RMCTSEngine:
    """
    Paper-aligned MCTS: rollouts = Select + Expand(Alg.2) + optional greedy Simulate
    + Backprop along selection path + new edge + sim prefix.
    """

    def __init__(
        self,
        file_path: str,
        baseline_r0: str,
        verifier: CRustVerifier,
        llm: LLMRefiner,
        require_tests: bool,
        hyp: MCTSHyper,
    ) -> None:
        self.file_path = file_path
        self.baseline_r0 = baseline_r0
        self.verifier = verifier
        self.llm = llm
        self.require_tests = require_tests
        self.hyp = hyp
        self.baseline_total = max(1, baseline_total_from_code(baseline_r0))
        b_out = verify_rust_file(verifier, file_path, baseline_r0, require_tests=require_tests)
        self.s0, _ = _snap(
            baseline_r0, baseline_total=self.baseline_total, compilable=b_out.compilable
        )
        self.c0 = compile_score_C(b_out.n_compile_errors)
        self.root = MCTSNode(
            kind="INIT",
            code=baseline_r0,
            meta={"s0": self.s0, "c0": self.c0, "b_total": self.baseline_total},
        )
        self.best_success: Optional[MCTSNode] = None
        self.best_s: float = -1.0

    def mcts_search(self) -> MCTSNode:
        for _ in range(self.hyp.n_rollouts):
            self._mcts_rollout()
        self.root.meta["best_success"] = self.best_success
        self.root.meta["best_s"] = self.best_s
        return self.root

    def _mcts_rollout(self) -> None:
        path_s = _select_to_leaf(self.root, self.hyp.uct_c)
        leaf = path_s[-1]

        if _is_terminal(leaf) or leaf.depth >= self.hyp.max_depth:
            _backprop(path_s, 0.0)
            return

        ex_path, r_expand, r_sim = self._expand(leaf, path_s)
        if not ex_path:
            _backprop(path_s, 0.0)
            return
        _backprop(ex_path, r_expand + r_sim)

    def _expand(
        self,
        node: MCTSNode,
        path_s: List[MCTSNode],
    ) -> tuple[List[MCTSNode], float, float]:
        """
        Returns (path for backprop, r_expand, r_sim).
        backprop path = path_s + [new child] + [sim virtual nodes] (sim nodes are still MCTSNode for book-keeping, linked under child as meta-only chain — we attach sim chain under child in meta to avoid tree pollution).
        """
        fp = self.file_path
        hyp = self.hyp
        btot = self.baseline_total
        s0, c0 = self.s0, self.c0
        base = self.baseline_r0

        if node.kind == "INIT" and not node.expanded:
            best_pair: list[tuple[MCTSNode, float]] = []
            for v in range(hyp.root_branching):
                code = self.llm.initial_gen(base, v)
                o = verify_rust_file(self.verifier, fp, code, require_tests=self.require_tests)
                S, _ = _snap(code, baseline_total=btot, compilable=o.compilable)
                c1 = compile_score_C(o.n_compile_errors)
                r = node_reward_R(c0, c1, s0, S, w=hyp.w_reward)
                sess = self.llm.session_after_initial(base, code)
                kind: NodeKind = "SUCCESS" if o.success else ("GEN" if o.compilable else "FIX")
                child = MCTSNode(
                    kind=kind, code=code, parent=node, s_val=S, c_val=c1, session=sess, meta={}
                )
                node.children.append(child)
                best_pair.append((child, r))
                if kind == "SUCCESS" and (self.best_success is None or S > self.best_s):
                    self.best_success, self.best_s = child, S
            node.expanded = True
            if not node.children:
                return [], 0.0, 0.0
            ch, r_ch = max(best_pair, key=lambda t: t[0].s_val)
            sim_r = self._greedy_simulate(ch)
            return path_s + [ch], r_ch, sim_r

        if node.session is None:
            return [], 0.0, 0.0

        o_leaf = verify_rust_file(self.verifier, fp, node.code, require_tests=self.require_tests)
        fb, kind_fb = _feedback_str(o_leaf)
        if not fb:
            fb = "Refine for LAC2R: fewer RPC/RPR/LUC/UCE/UTC; keep tests green."
        s_prev, c_prev = node.s_val, node.c_val
        use_no_fb = (random.random() < hyp.p_no_feedback_expand) and (node.kind == "GEN")
        if use_no_fb:
            sess2 = _copy_session(node)
            new_code = self.llm.continue_refinement_without_feedback(sess2, node.code, variant=0)
        else:
            sess2 = _copy_session(node)
            new_code = self.llm.fix_with_feedback(sess2, feedback=fb, err_kind=kind_fb)

        o2 = verify_rust_file(self.verifier, fp, new_code, require_tests=self.require_tests)
        S, _ = _snap(new_code, baseline_total=btot, compilable=o2.compilable)
        c1 = compile_score_C(o2.n_compile_errors)
        r = node_reward_R(c_prev, c1, s_prev, S, w=hyp.w_reward)
        nkind: NodeKind = "SUCCESS" if o2.success else ("FIX" if (not o2.compilable or not o2.success) else "GEN")
        child = MCTSNode(
            kind=nkind, code=new_code, parent=node, s_val=S, c_val=c1, session=sess2, meta={}
        )
        node.children.append(child)
        node.expanded = True
        if nkind == "SUCCESS" and (self.best_success is None or S > self.best_s):
            self.best_success, self.best_s = child, S
        sim_r = self._greedy_simulate(child)
        return path_s + [child], r, sim_r

    def _greedy_simulate(self, start: MCTSNode) -> float:
        if self.hyp.simulation_depth <= 0:
            return 0.0
        r_acc = 0.0
        cur_code = start.code
        s_prev, c_prev = start.s_val, start.c_val
        sess = _copy_session(start) if start.session else self.llm.session_after_initial(self.baseline_r0, cur_code)
        for _k in range(self.hyp.simulation_depth):
            o = verify_rust_file(self.verifier, self.file_path, cur_code, require_tests=self.require_tests)
            if o.success:
                break
            fb, kfb = _feedback_str(o)
            nxt = self.llm.fix_with_feedback(sess, feedback=fb, err_kind=kfb)
            o2 = verify_rust_file(self.verifier, self.file_path, nxt, require_tests=self.require_tests)
            S, _ = _snap(nxt, baseline_total=self.baseline_total, compilable=o2.compilable)
            c1 = compile_score_C(o2.n_compile_errors)
            r = node_reward_R(c_prev, c1, s_prev, S, w=self.hyp.w_reward)
            r_acc += r
            cur_code, s_prev, c_prev = nxt, S, c1
            if o2.success:
                break
        return r_acc


def run_mcts(
    file_path: str,
    baseline_r0: str,
    verifier: CRustVerifier,
    llm: LLMRefiner,
    require_tests: bool,
    hyp: MCTSHyper,
) -> MCTSNode:
    return LAC2RMCTSEngine(
        file_path, baseline_r0, verifier, llm, require_tests, hyp
    ).mcts_search()
