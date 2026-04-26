"""
Heterogeneous + tiered LLM refinement (paper §2.3); RAG-injected repair prompts.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import List, Optional

from .rag import RagContext, stackoverflow_rag

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover
    OpenAI = None  # type: ignore

FUNC_RE = re.compile(r"<FUNC>([\s\S]*?)</FUNC>", re.IGNORECASE)


def _extract_func(text: str) -> str:
    m = FUNC_RE.search(text)
    if m:
        return m.group(1).strip()
    m3 = re.search(r"```rust\s*([\s\S]*?)```", text)
    if m3:
        return m3.group(1).strip()
    m4 = re.search(r"```\s*([\s\S]*?)```", text)
    if m4:
        return m4.group(1).strip()
    return text.strip()


@dataclass
class LAC2RLLMConfig:
    model_cheap: str = "gpt-4o-mini"
    model_strong: str = "gpt-4o"
    model_alt: str = "gpt-4o"
    use_rag: bool = True
    tiered: bool = True
    max_tokens: int = 4096
    temperature: float = 0.2

    @classmethod
    def from_env(cls) -> "LAC2RLLMConfig":
        return cls(
            model_cheap=os.getenv("LAC2R_MODEL_CHEAP", "gpt-4o-mini"),
            model_strong=os.getenv("LAC2R_MODEL_STRONG", "gpt-4o"),
            model_alt=os.getenv("LAC2R_MODEL_ALT", os.getenv("LAC2R_MODEL_STRONG", "gpt-4o")),
            use_rag=os.getenv("LAC2R_RAG", "1") not in ("0", "false", "no"),
            tiered=os.getenv("LAC2R_TIERED", "1") not in ("0", "false", "no"),
        )


@dataclass
class RefineSession:
    system: str = (
        "You are an expert Rust engineer performing C2Rust de-unsafing. "
        "Respect the crate's public API. Use <FUNC> </FUNC> around the code."
    )
    messages: List[dict] = field(default_factory=list)

    def add_user(self, text: str) -> None:
        self.messages.append({"role": "user", "content": text})

    def add_assistant(self, text: str) -> None:
        self.messages.append({"role": "assistant", "content": text})


class LLMRefiner:
    def __init__(self, config: LAC2RLLMConfig | None = None):
        self.cfg = config or LAC2RLLMConfig.from_env()
        self._client: Optional[OpenAI] = None
        if OpenAI and os.getenv("OPENAI_API_KEY"):
            self._client = OpenAI()

    @property
    def available(self) -> bool:
        return self._client is not None

    def _complete(self, model: str, session: RefineSession) -> str:
        if not self._client:
            return self._mock_refine(
                session.messages[-1]["content"] if session.messages else ""
            )
        rsp = self._client.chat.completions.create(
            model=model,
            temperature=self.cfg.temperature,
            max_tokens=self.cfg.max_tokens,
            messages=[{"role": "system", "content": session.system}, *session.messages],
        )
        ch = rsp.choices[0].message
        return (ch.content or "").strip()

    @staticmethod
    def _mock_refine(user_prompt: str) -> str:
        """
        Offline fallback when OPENAI_API_KEY is unset: return the first ```rust` block
        (or a minimal stub) with `unsafe` tokens removed, wrapped in <FUNC>.
        """
        m = re.search(r"```rust\s*([\s\S]*?)```", user_prompt)
        body = m.group(1).strip() if m else (user_prompt[:2000] if user_prompt else "pub fn _mock() {}")
        body = body.replace("unsafe ", "")
        return f"<FUNC>\n{body}\n</FUNC>"

    def build_main_user_prompt(
        self,
        unsafe_rust: str,
        *,
        call_sites: str = "",
        imports: str = "",
        globals: str = "",
        rag: Optional[RagContext] = None,
    ) -> str:
        rag_block = ""
        if rag and rag.snippets:
            rag_block = "External hints:\n- " + "\n- ".join(rag.snippets[:5]) + "\n\n"
        return (
            f"Refine this Rust. Put code only inside <FUNC> </FUNC>.\n{rag_block}"
            f"```rust\n{unsafe_rust}\n```\n"
            f"Call sites: {call_sites or 'N/A'}\n"
            f"Imports: {imports or 'N/A'}\n"
            f"Globals: {globals or 'N/A'}\n"
        )

    def _root_model(self, variant: int) -> str:
        # Paper: 2 from one LLM, 2 from another — alternate cheap/strong vs alt
        if variant in (0, 1):
            return self.cfg.model_strong if variant == 0 else self.cfg.model_cheap
        return self.cfg.model_alt

    def initial_gen(self, unsafe_rust: str, variant: int) -> str:
        session = RefineSession()
        session.add_user(self.build_main_user_prompt(unsafe_rust))
        text = self._complete(self._root_model(variant), session)
        return _extract_func(text)

    def new_session_from_seed(self, unsafe_rust: str) -> RefineSession:
        s = RefineSession()
        s.add_user(self.build_main_user_prompt(unsafe_rust))
        return s

    def session_after_initial(self, unsafe_rust: str, generated_code: str) -> RefineSession:
        s = self.new_session_from_seed(unsafe_rust)
        s.add_assistant(generated_code)
        return s

    def fix_with_feedback(
        self,
        session: RefineSession,
        *,
        feedback: str,
        err_kind: str,
        use_rag: bool = True,
    ) -> str:
        extra = f"[{err_kind}]\n{feedback}\n"
        if use_rag and self.cfg.use_rag:
            rag = stackoverflow_rag(feedback, language="rust")
            if rag and rag.snippets:
                extra += "RAG: " + "; ".join(rag.snippets[:3]) + "\n"
        session.add_user("Fix. " + extra + " Full function(s) in <FUNC> </FUNC> only.")
        # Escalation: later attempts use the stronger model when tiered
        use_strong = self.cfg.tiered and (len(session.messages) >= 3)
        model = self.cfg.model_strong if use_strong else self.cfg.model_cheap
        text = self._complete(model, session)
        out = _extract_func(text)
        session.add_assistant(out)
        return out

    def continue_refinement_without_feedback(
        self, session: RefineSession, current_code: str, *, variant: int = 0
    ) -> str:
        """
        Paper: action (null, LLM_k) — refinement without explicit validator string.
        Drives diversity when compile/test feedback is uninformative.
        """
        session.add_user(
            "Propose a safer, more idiomatic version of the Rust above without repeating "
            "the same structure. Use <FUNC> </FUNC> only. Keep public API and semantics."
        )
        model = self.cfg.model_cheap if variant % 2 == 0 else self.cfg.model_strong
        text = self._complete(model, session)
        out = _extract_func(text)
        session.add_assistant(out)
        return out
