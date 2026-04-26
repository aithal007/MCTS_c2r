"""
Demo CLI: LAC2R MCTS refine on a file in the dummy Cargo workspace.
Usage (from repository root):
  set PYTHONPATH=.
  python run_lac2r_demo.py
"""

import os
import sys

# Ensure package root on path
_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from crust_env.lac2r.service import LAC2RConfig, run_lac2r_refine  # noqa: E402


def main() -> None:
    workspace = os.path.join(_ROOT, "crust_env", "dummy_workspace")
    cfg = LAC2RConfig(
        workspace=workspace,
        file_path="src/math_ops.rs",
        n_rollouts=3,
        max_depth=3,
        root_branching=2,
        require_tests=True,
        write_best=False,
    )
    out = run_lac2r_refine(cfg)
    print(out)


if __name__ == "__main__":
    main()
