from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class Vocab:
    stoi: dict[str, int]
    itos: list[str]
    pad_token: str = "<PAD>"
    unk_token: str = "<UNK>"

    @property
    def pad_id(self) -> int:
        return self.stoi[self.pad_token]

    @property
    def unk_id(self) -> int:
        return self.stoi[self.unk_token]

    def encode(self, tokens: Iterable[str]) -> list[int]:
        unk = self.unk_id
        return [self.stoi.get(t, unk) for t in tokens]

    def decode(self, ids: Iterable[int]) -> str:
        out: list[str] = []
        for i in ids:
            if 0 <= i < len(self.itos):
                out.append(self.itos[i])
        return "".join(out)

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)

    @staticmethod
    def from_json(s: str) -> "Vocab":
        obj = json.loads(s)
        return Vocab(**obj)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json(), encoding="utf-8")

    @staticmethod
    def load(path: str | Path) -> "Vocab":
        return Vocab.from_json(Path(path).read_text(encoding="utf-8"))


def build_vocab(
    tokenized_texts: Iterable[Iterable[str]],
    max_size: int = 5000,
    min_freq: int = 1,
    pad_token: str = "<PAD>",
    unk_token: str = "<UNK>",
) -> Vocab:
    freq: dict[str, int] = {}
    for tokens in tokenized_texts:
        for t in tokens:
            freq[t] = freq.get(t, 0) + 1

    items = [(t, c) for t, c in freq.items() if c >= min_freq]
    items.sort(key=lambda x: (-x[1], x[0]))
    tokens = [t for t, _ in items[: max(0, max_size - 2)]]

    itos = [pad_token, unk_token] + tokens
    stoi = {t: i for i, t in enumerate(itos)}
    return Vocab(stoi=stoi, itos=itos, pad_token=pad_token, unk_token=unk_token)

