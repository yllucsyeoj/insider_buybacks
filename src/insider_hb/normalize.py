"""Canonicalize free-text Form 4 officer titles into coarse role labels. Pure."""
from __future__ import annotations

import re

# Order matters: more specific patterns first.
_ROLE_MAP: list[tuple[str, str]] = [
    (r"chief executive|\bceo\b", "CEO"),
    (r"chief financial|principal financial|\bcfo\b", "CFO"),
    (r"chief operating|\bcoo\b", "COO"),
    (r"chief technology|\bcto\b", "CTO"),
    (r"principal accounting|chief accounting", "CAO"),
    (r"general counsel|chief legal", "GC"),
    (r"executive chair", "Executive Chair"),
    (r"(?<!vice )\bpresident\b", "President"),
]


def normalize_title(title: str) -> str:
    t = re.sub(r"\s+", " ", (title or "").strip()).lower()
    if not t:
        return ""
    for pattern, label in _ROLE_MAP:
        if re.search(pattern, t):
            return label
    return "Officer"
