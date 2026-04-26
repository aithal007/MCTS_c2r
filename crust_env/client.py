"""
CRust OpenEnv Client

Provides a typed HTTP client for interacting with the CRust Migration environment.
Compatible with the openenv-core EnvClient interface.

Usage:
    from crust_env.client import CRustClient

    client = CRustClient(base_url="https://adithyakommuri-meta-hackathon-final.hf.space")
    obs   = client.reset(phase=1, constraints=["Do not use the unsafe keyword"])
    result = client.step(file_path="src/math_ops.rs", code_content="pub fn add...")
    state  = client.state()
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import requests


# ── Typed data models ─────────────────────────────────────────────────────────

@dataclass
class CRustObservation:
    current_target: Optional[str]
    c_source_code: str
    constraints: List[str]
    recent_errors: List[Dict[str, Any]]
    dependency_context: Dict[str, str]
    phase: int
    files_remaining: int
    step: int


@dataclass
class CRustStepResult:
    observation: CRustObservation
    reward: float
    done: bool
    info: Dict[str, Any]


@dataclass
class CRustState:
    status: str
    phase: int
    schedule: List[str]
    current_idx: int
    files_total: int
    files_done: int
    step_count: int
    constraints: List[str]
    session_id: str
    extra: Dict[str, Any] = field(default_factory=dict)


# ── Client ────────────────────────────────────────────────────────────────────

class CRustClient:
    """
    HTTP client for the CRust C-to-Rust Migration OpenEnv environment.

    Compatible with the openenv-core client conventions:
      - reset()  → CRustObservation
      - step()   → CRustStepResult
      - state()  → CRustState
    """

    def __init__(
        self,
        base_url: str = "https://adithyakommuri-meta-hackathon-final.hf.space",
        timeout: int = 60,
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout  = timeout
        self.session  = requests.Session()

    # ── OpenEnv interface ──────────────────────────────────────────────────

    def reset(
        self,
        phase: int = 1,
        constraints: Optional[List[str]] = None,
    ) -> CRustObservation:
        """Reset the environment and return the first observation."""
        payload: Dict[str, Any] = {"phase": phase}
        if constraints is not None:
            payload["constraints"] = constraints

        resp = self.session.post(
            f"{self.base_url}/reset",
            json=payload,
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return self._parse_observation(resp.json())

    def step(
        self,
        file_path: str,
        code_content: str,
    ) -> CRustStepResult:
        """Submit a Rust translation and receive a reward."""
        resp = self.session.post(
            f"{self.base_url}/step",
            json={"file_path": file_path, "code_content": code_content},
            timeout=self.timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        return CRustStepResult(
            observation=self._parse_observation(data.get("observation", {})),
            reward=float(data.get("reward", 0.0)),
            done=bool(data.get("done", False)),
            info=data.get("info", {}),
        )

    def state(self) -> CRustState:
        """Return the full internal environment state."""
        resp = self.session.get(f"{self.base_url}/state", timeout=self.timeout)
        resp.raise_for_status()
        d = resp.json()
        return CRustState(
            status=d.get("status", "unknown"),
            phase=d.get("phase", 1),
            schedule=d.get("schedule", []),
            current_idx=d.get("current_idx", 0),
            files_total=d.get("files_total", 0),
            files_done=d.get("files_done", 0),
            step_count=d.get("step_count", 0),
            constraints=d.get("constraints", []),
            session_id=d.get("session_id", ""),
            extra={k: v for k, v in d.items()
                   if k not in {"status","phase","schedule","current_idx",
                                "files_total","files_done","step_count",
                                "constraints","session_id"}},
        )

    def observation(self) -> CRustObservation:
        """Return the current agent-visible partial observation."""
        resp = self.session.get(f"{self.base_url}/observation", timeout=self.timeout)
        resp.raise_for_status()
        return self._parse_observation(resp.json())

    def health(self) -> Dict[str, Any]:
        """Health check."""
        resp = self.session.get(f"{self.base_url}/health", timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    # ── Convenience factory ────────────────────────────────────────────────

    @classmethod
    def from_hub(cls, repo_id: str = "Adithyakommuri/meta_hackathon_final") -> "CRustClient":
        """Connect to the environment hosted on Hugging Face Spaces."""
        username, space_name = repo_id.split("/")
        url = f"https://{username.lower()}-{space_name.lower().replace('_', '-')}.hf.space"
        return cls(base_url=url)

    # ── Private helpers ────────────────────────────────────────────────────

    @staticmethod
    def _parse_observation(data: Dict[str, Any]) -> CRustObservation:
        return CRustObservation(
            current_target=data.get("current_target"),
            c_source_code=data.get("c_source_code", ""),
            constraints=data.get("constraints", []),
            recent_errors=data.get("recent_errors", []),
            dependency_context=data.get("dependency_context", {}),
            phase=data.get("phase", 1),
            files_remaining=data.get("files_remaining", 0),
            step=data.get("step", 0),
        )

    def close(self):
        self.session.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
