from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
from torch.utils.data import Dataset

from .text import tokenize_char, truncate
from .vocab import Vocab


@dataclass(frozen=True)
class Sample:
    text: str
    label: int


def load_csv(path: str | Path) -> list[Sample]:
    path = Path(path)
    rows: list[Sample] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError("CSV must have headers: text,label")
        for r in reader:
            text = (r.get("text") or "").strip()
            label_raw = (r.get("label") or "").strip()
            if not text or label_raw == "":
                continue
            label = int(label_raw)
            if label not in (0, 1):
                raise ValueError(f"label must be 0/1, got {label}")
            rows.append(Sample(text=text, label=label))
    return rows


class TextClsDataset(Dataset):
    def __init__(self, samples: list[Sample], vocab: Vocab, max_len: int) -> None:
        self.samples = samples
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[list[int], int]:
        s = self.samples[idx]
        tokens = truncate(tokenize_char(s.text), self.max_len)
        ids = self.vocab.encode(tokens)
        return ids, s.label


def collate_batch(
    batch: Iterable[tuple[list[int], int]],
    pad_id: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    ids_list: list[list[int]] = []
    labels_list: list[int] = []
    lengths_list: list[int] = []
    for ids, label in batch:
        ids_list.append(ids)
        labels_list.append(int(label))
        lengths_list.append(len(ids))

    max_len = max(lengths_list) if lengths_list else 0
    x = torch.full((len(ids_list), max_len), pad_id, dtype=torch.long)
    for i, ids in enumerate(ids_list):
        if ids:
            x[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)
    lengths = torch.tensor(lengths_list, dtype=torch.long)
    y = torch.tensor(labels_list, dtype=torch.long)
    return x, lengths, y

