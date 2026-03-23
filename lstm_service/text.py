from __future__ import annotations

import re
from typing import Iterable


_space_re = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    return _space_re.sub(" ", text.strip())


def tokenize_char(text: str) -> list[str]:
    text = normalize_text(text)
    return list(text)


def truncate(tokens: Iterable[str], max_len: int) -> list[str]:
    out: list[str] = []
    for t in tokens:
        out.append(t)
        if len(out) >= max_len:
            break
    return out

