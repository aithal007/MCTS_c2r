"""
Heuristic counters for the five unsafe-construct families (LAC2R, paper Eq. 3):
  RPC, RPR, LUC, UCE, UTC
"""

from __future__ import annotations

import re
from dataclasses import dataclass, fields


@dataclass
class UnsafeConstructCounts:
    rpc: int = 0
    rpr: int = 0
    luc: int = 0
    uce: int = 0
    utc: int = 0

    def total(self) -> int:
        return self.rpc + self.rpr + self.luc + self.uce + self.utc

    def as_dict(self) -> dict:
        return {f.name: getattr(self, f.name) for f in fields(self)}


def _strip_comments(src: str) -> str:
    s = re.sub(r"//[^\n]*", " ", src)
    s = re.sub(r"/\*.*?\*/", " ", s, flags=re.DOTALL)
    return s


def _unsafe_block_spans(text: str) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    i = 0
    while i < len(text):
        m = re.search(r"\bunsafe\s*\{", text[i:])
        if not m:
            break
        start = i + m.start()
        j = i + m.end()
        depth = 1
        k = j
        while k < len(text) and depth:
            if text[k] == "{":
                depth += 1
            elif text[k] == "}":
                depth -= 1
            k += 1
        if depth == 0:
            spans.append((start, k))
        i = j
    return spans


def _lines_touching_spans(text: str, spans: list[tuple[int, int]]) -> int:
    if not spans:
        return 0
    n = 0
    for line in text.splitlines():
        mid = text.find(line)
        if mid < 0:
            continue
        # Any overlap between line char range and a span
        le = mid + len(line)
        if any(s < le and e > mid for s, e in spans):
            n += 1
    return n


def count_unsafe_constructs(code: str) -> UnsafeConstructCounts:
    c = _strip_comments(code)
    out = UnsafeConstructCounts()

    # RPC — raw pointer declarations in type position
    out.rpc = len(re.findall(r"\*const(?:\b|\s)", c)) + len(re.findall(r"\*mut(?:\b|\s)", c))

    # RPR — raw deref and ptr helpers
    deref = len(
        re.findall(
            r"(?<![\*'\"/])\*(?=\s*[\(a-zA-Z_])"  # *foo or *(
            r"|(?<=\s)\*(?=\s*[\(a-zA-Z_])",
            c,
        )
    )
    out.rpr = deref + c.count("ptr::read(") + c.count("ptr::write(")

    spans = _unsafe_block_spans(c)
    out.luc = 0
    for s, e in spans:
        block = c[s:e]
        out.luc += max(0, block.count("\n") + 1)

    uce = 0
    for s, e in spans:
        chunk = c[s:e]
        uce += len(
            re.findall(
                r"\b[A-Za-z_][A-Za-z0-9_]*\s*!(?:::\[[^\]]*\])?\s*\("
                r"|[A-Za-z_][A-Za-z0-9_]*\s*::[A-Za-z_][A-Za-z0-9_]*\s*\("
                r"|[A-Za-z_][A-Za-z0-9_]*\s*\(",
                chunk,
            )
        )
    out.uce = uce

    out.utc = len(
        re.findall(
            r"\bstd::mem::transmute\b|\bcore::mem::transmute\b"
            r"|as\s+\*const|as\s+\*mut|as\s*\(\s*\*const|as\s*\(\s*\*mut",
            c,
        )
    )
    return out
