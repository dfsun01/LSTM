from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import torch

from .config import ModelConfig, TextConfig
from .model import LSTMClassifier
from .vocab import Vocab


def save_artifacts(
    out_dir: str | Path,
    model: LSTMClassifier,
    vocab: Vocab,
    model_cfg: ModelConfig,
    text_cfg: TextConfig,
) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    vocab.save(out_dir / "vocab.json")
    (out_dir / "config.json").write_text(
        json.dumps(
            {"model": asdict(model_cfg), "text": asdict(text_cfg)},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    torch.save(model.state_dict(), out_dir / "model.pt")


def load_artifacts(
    dir_path: str | Path,
    device: str | torch.device = "cpu",
) -> tuple[LSTMClassifier, Vocab, ModelConfig, TextConfig]:
    dir_path = Path(dir_path)
    vocab = Vocab.load(dir_path / "vocab.json")
    cfg_obj = json.loads((dir_path / "config.json").read_text(encoding="utf-8"))
    model_cfg = ModelConfig(**cfg_obj["model"])
    text_cfg = TextConfig(**cfg_obj["text"])

    model = LSTMClassifier(model_cfg)
    sd = torch.load(dir_path / "model.pt", map_location=device)
    model.load_state_dict(sd)
    model.to(device)
    model.eval()
    return model, vocab, model_cfg, text_cfg

