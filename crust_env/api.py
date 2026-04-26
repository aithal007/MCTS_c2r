"""
CRust Migration OpenEnv — api.py

FastAPI application exposing the CRust environment via the OpenEnv standard interface,
plus a /train/* namespace that runs GRPO training directly on the Space's A10G GPU.

OpenEnv endpoints:
  POST /reset          — Start a new episode
  POST /step           — Submit a Rust translation action
  GET  /state          — Full internal state (debug)
  GET  /observation    — Agent-visible partial observation
  GET  /health         — Health check
  GET  /info           — Project and environment metadata

Training endpoints (runs on the Space GPU):
  POST /train/start    — Launch GRPO training in a background thread
  GET  /train/status   — Live progress: step, reward, log tail
  POST /train/stop     — Gracefully signal stop
"""

import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional

from .env import MigrationEnv
from . import trainer_daemon
from .lac2r.service import LAC2RConfig, run_lac2r_refine

# ── App setup ──────────────────────────────────────────────────────────────

app = FastAPI(
    title="CRust Migration OpenEnv",
    description=(
        "An OpenEnv-compatible RL environment for training LLMs to migrate "
        "legacy C codebases to memory-safe, modular Rust. "
        "Built for the Meta OpenEnv Hackathon — Theme #2: Long-Horizon Planning. "
        "<br><br>"
        "<b>Training is live on this Space's A10G GPU</b> — "
        "POST /train/start to kick it off, GET /train/status to watch rewards climb."
    ),
    version="2.0.0",
)

# ── Environment init ────────────────────────────────────────────────────────

WORKSPACE_DIR = os.getenv(
    "CRUST_WORKSPACE",
    os.path.join(os.path.dirname(__file__), "dummy_workspace")
)

crust_env = MigrationEnv(workspace_dir=WORKSPACE_DIR)

# ── Pydantic models ─────────────────────────────────────────────────────────

class ResetRequest(BaseModel):
    constraints: Optional[List[str]] = Field(
        default=None,
        description=(
            "Multi-constraint directives injected at episode start. "
            "E.g. ['Do not use the unsafe keyword', 'Maintain a CBO score below 3']. "
            "Defaults to the standard CRust constraints if not provided."
        )
    )
    phase: int = Field(
        default=1, ge=1, le=4,
        description=(
            "Curriculum phase controlling migration scope. "
            "1=single leaf node, 2=two-file chain, "
            "3=partial DAG, 4=full repository."
        )
    )


class StepAction(BaseModel):
    file_path: str = Field(
        description="Relative path for the Rust file to write (e.g. 'src/math_ops.rs')"
    )
    code_content: str = Field(
        description="Complete Rust source code to write and verify."
    )


class LAC2RRequest(BaseModel):
    file_path: str = Field(
        default="src/math_ops.rs",
        description="Path relative to the Cargo workspace (e.g. src/math_ops.rs).",
    )
    n_rollouts: int = Field(default=10, ge=1, le=200)
    max_depth: int = Field(default=5, ge=1, le=20)
    uct_c: float = Field(default=1.5, ge=0.0, le=10.0)
    w_reward: float = Field(default=2.0, ge=0.0, le=20.0)
    root_branching: int = Field(default=4, ge=1, le=8)
    simulation_depth: int = Field(
        default=0, ge=0, le=10,
        description="Greedy virtual sim steps after expand (paper Rollout; 0=off to save LLM).",
    )
    p_no_feedback_expand: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="Prob. of (null, LLM) no-feedback child on GEN (paper §2.3).",
    )
    require_tests: bool = Field(default=True)
    write_best: bool = Field(
        default=False,
        description="If true, writes the best SUCCESS candidate over the file (use with care).",
    )


class TrainRequest(BaseModel):
    max_steps: int = Field(
        default=100, ge=10, le=500,
        description="Number of GRPO training steps. 100 ≈ 20 min on A10G."
    )
    model_name: str = Field(
        default="Qwen/Qwen2.5-3B-Instruct",
        description="HF model ID to fine-tune. Qwen2.5-3B in bf16 fits comfortably on A10G Small (no quantization needed)."
    )
    hf_token: str = Field(
        default="",
        description=(
            "Your HF write token (from https://huggingface.co/settings/tokens). "
            "If blank, reads the HF_TOKEN environment variable (preferred — set it as a Space Secret)."
        )
    )
    hf_repo: str = Field(
        default="crust-grpo-qwen25-3b",
        description="HF Hub repo name (under your username) to push the trained LoRA."
    )
    phase: int = Field(
        default=1, ge=1, le=4,
        description="Curriculum phase for training data generation."
    )


# ── OpenEnv endpoints ────────────────────────────────────────────────────────

@app.post("/reset", summary="Reset the environment for a new episode")
def reset_env(request: ResetRequest = ResetRequest()) -> Dict[str, Any]:
    """
    Resets the CRust environment.
    Parses the legacy C dependency graph, selects the first migration target
    based on the curriculum phase, and returns the initial observation.
    """
    return crust_env.reset(constraints=request.constraints, phase=request.phase)


@app.post("/step", summary="Submit a Rust translation and receive a reward")
def step_env(action: StepAction) -> Dict[str, Any]:
    """
    Submit a Rust translation for verification.

    Pipeline:
    1. Write code to sandboxed Cargo workspace
    2. `cargo check` — compilation (0.30 weight)
    3. `cargo test`  — semantic equivalence (0.30 weight)
    4. CBO / LCOM metrics (0.20 weight)
    5. Memory-safety check — no `unsafe` (0.20 weight)

    Reward: [0.01, 0.99]
    """
    return crust_env.step(action.model_dump())


@app.get("/state", summary="Full internal state (debugging)")
def get_state() -> Dict[str, Any]:
    return crust_env.state


@app.get("/observation", summary="Agent-visible partial observation")
def get_observation() -> Dict[str, Any]:
    return crust_env.observation()


@app.get("/health", summary="Health check")
def health_check() -> Dict[str, Any]:
    return {
        "status": "healthy",
        "workspace": WORKSPACE_DIR,
        "workspace_exists": os.path.isdir(WORKSPACE_DIR),
        "training_status": trainer_daemon.get_status()["status"],
    }


@app.post(
    "/lac2r/refine",
    summary="LAC2R — MCTS-guided multi-trajectory safe Rust refinement (paper: arXiv:2505.15858)",
)
def lac2r_refine(request: LAC2RRequest = LAC2RRequest()) -> Dict[str, Any]:
    """
    Runs MCTS (UCT, heterogeneous/tiered LLM, optional RAG on repair) on one workspace file.
    Set ``OPENAI_API_KEY`` for real LLM calls; otherwise a deterministic offline mock is used.
    """
    cfg = LAC2RConfig(
        workspace=WORKSPACE_DIR,
        file_path=request.file_path,
        n_rollouts=request.n_rollouts,
        max_depth=request.max_depth,
        uct_c=request.uct_c,
        w_reward=request.w_reward,
        root_branching=request.root_branching,
        simulation_depth=request.simulation_depth,
        p_no_feedback_expand=request.p_no_feedback_expand,
        require_tests=request.require_tests,
        write_best=request.write_best,
    )
    return run_lac2r_refine(cfg)


@app.get("/info", summary="Environment metadata")
def env_info() -> Dict[str, Any]:
    return {
        "project": "CRust — C-to-Rust RL Migration Environment",
        "hackathon": "Meta OpenEnv Hackathon — Theme #2: Long-Horizon Planning",
        "interface": "OpenEnv (reset / step / state / observation) + POST /lac2r/refine (LAC2R MCTS)",
        "training_stack": "TRL GRPO + PEFT LoRA + FastAPI (runs on this Space's A10G GPU)",
        "default_constraints": [
            "Do not use the unsafe keyword",
            "Maintain a CBO score below 3",
        ],
        "curriculum_phases": {
            1: "Single leaf node (isolated function translation)",
            2: "Two-file dependency chain",
            3: "Partial DAG (cross-file with headers)",
            4: "Full repository (long-horizon cascading error resolution)",
        },
        "reward_components": {
            "compilation":   0.30,
            "tests":         0.30,
            "memory_safety": 0.20,
            "cbo_constraint":0.10,
            "cohesion_lcom": 0.10,
        },
        "anti_hacking_measures": [
            "Protected test files (agent cannot modify cargo tests)",
            "Strict subprocess timeouts (30s compilation, 60s tests)",
            "Path traversal protection on all file writes",
            "unsafe keyword penalty (−0.50)",
            "CBO violation penalty (−0.20)",
            "Max step limit (200 steps per episode)",
        ],
        "lac2r_rl_shaping": {
            "env_var_enable": "CRUST_LAC2R_REWARD=1",
            "weight_var": "CRUST_LAC2R_WEIGHT (default 0.06)",
            "effect": "Adds LAC2R safety S(r) bonus to env.step() reward (paper Eq. 3).",
        },
    }


# ── Training endpoints ────────────────────────────────────────────────────────

@app.post("/train/start", summary="Launch GRPO training on the Space GPU")
def train_start(request: TrainRequest = TrainRequest()) -> Dict[str, Any]:
    """
    Launches GRPO fine-tuning of the LLM in a **background thread** on the A10G GPU.

    The OpenEnv endpoints (/reset, /step) keep serving concurrently.

    Tips:
    - Set your HF token as a **Space Secret** named `HF_TOKEN` — safer than passing it here.
    - 100 steps ≈ 20 min on A10G Small with Qwen2.5-3B.
    - Poll GET /train/status to watch rewards improve.
    """
    if trainer_daemon.is_running():
        raise HTTPException(
            status_code=409,
            detail="Training is already running. Call POST /train/stop first, or poll GET /train/status."
        )

    # Prefer Space Secret over request body
    token = request.hf_token or os.getenv("HF_TOKEN", "")
    username = os.getenv("HF_USERNAME", "Adithyakommuri")
    repo = f"{username}/{request.hf_repo}"

    ok, msg = trainer_daemon.start_training(
        max_steps  = request.max_steps,
        model_name = request.model_name,
        hf_token   = token,
        hf_repo    = repo,
        workspace  = WORKSPACE_DIR,
        phase      = request.phase,
    )

    if not ok:
        raise HTTPException(status_code=500, detail=msg)

    return {
        "message":    msg,
        "model":      request.model_name,
        "max_steps":  request.max_steps,
        "phase":      request.phase,
        "push_to":    f"https://huggingface.co/{repo}" if token else "(no token — will save locally to /app/crust_lora)",
        "monitor":    "GET /train/status",
    }


@app.get("/train/status", summary="Live training progress")
def train_status() -> Dict[str, Any]:
    """
    Returns the current training progress.

    ```
    {
      "status":          "running",
      "step":            42,
      "max_steps":       100,
      "current_reward":  0.4312,
      "best_reward":     0.5100,
      "reward_history":  [0.12, 0.18, ...],
      "elapsed_seconds": 720,
      "gpu_name":        "NVIDIA A10G",
      "log":             ["Loading model...", "Step 10/100 reward=0.23", ...]
    }
    ```
    """
    s = trainer_daemon.get_status()
    return {
        **s,
        "progress_pct": round(100 * s["step"] / max(s["max_steps"], 1), 1),
        "eta_seconds":  (
            round((s["max_steps"] - s["step"]) * s["elapsed_seconds"] / max(s["step"], 1))
            if s["step"] > 0 and s["status"] == "running" else None
        ),
    }


@app.post("/train/stop", summary="Gracefully stop training")
def train_stop() -> Dict[str, Any]:
    """
    Signals the training thread to stop after the current step completes.
    The partially-trained model is still saved to /app/crust_lora.
    """
    if not trainer_daemon.is_running():
        return {"message": "No training is currently running."}
    trainer_daemon.request_stop()
    return {"message": "Stop signal sent — training will halt after the current step."}
