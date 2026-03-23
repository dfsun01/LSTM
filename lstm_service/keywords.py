from __future__ import annotations

from pathlib import Path


def load_keywords(path: str | Path) -> list[str]:
    p = Path(path)
    if not p.exists():
        return []
    words: list[str] = []
    for line in p.read_text(encoding="utf-8").splitlines():
        w = line.strip()
        if not w or w.startswith("#"):
            continue
        words.append(w)
    return words


def rule_hit(text: str, keywords: list[str]) -> bool:
    return any(w in text for w in keywords)

