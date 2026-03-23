from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int
    embed_dim: int = 64
    hidden_dim: int = 64
    num_layers: int = 1
    dropout: float = 0.1
    num_classes: int = 2


@dataclass(frozen=True)
class TextConfig:
    max_len: int = 200

