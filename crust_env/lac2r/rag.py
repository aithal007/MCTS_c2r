"""
Retrieval-augmented hints for error repair (stack overflow + optional web),
aligned with the RAG process diagram. Uses the public Stack Exchange API
(no key required for low volume); returns ranked text snippets.
"""

from __future__ import annotations

import re
import urllib.parse
from dataclasses import dataclass
from typing import List, Optional

import httpx


@dataclass
class RagContext:
    query: str
    snippets: List[str]
    sources: List[str]


def _sanitize(q: str) -> str:
    return re.sub(r"\s+", " ", q)[:200]


def stackoverflow_rag(
    issue_message: str,
    *,
    language: str = "rust",
    max_items: int = 3,
    timeout: float = 8.0,
) -> Optional[RagContext]:
    q = _sanitize(f"{language} {issue_message}")
    params = {
        "order": "desc",
        "sort": "relevance",
        "intitle": q,
        "site": "stackoverflow",
        "pagesize": str(max(1, max_items)),
    }
    url = "https://api.stackexchange.com/2.3/search?" + urllib.parse.urlencode(params)
    try:
        with httpx.Client(timeout=timeout) as client:
            r = client.get(url, headers={"User-Agent": "CRust-LAC2R/1.0"})
            r.raise_for_status()
            data = r.json()
    except Exception:
        return None
    items = data.get("items") or []
    snippets: list[str] = []
    links: list[str] = []
    for it in items[:max_items]:
        title = (it.get("title") or "").replace("&#39;", "'")
        link = it.get("link") or ""
        if title:
            snippets.append(title)
            links.append(link)
    if not snippets:
        return None
    return RagContext(query=q, snippets=snippets, sources=links)
