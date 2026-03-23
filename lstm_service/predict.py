from __future__ import annotations

import argparse

import torch

from .dataset import collate_batch
from .io import load_artifacts
from .text import tokenize_char, truncate


@torch.no_grad()
def predict_one(model, vocab, text_cfg, text: str, device: torch.device) -> tuple[int, float]:
    tokens = truncate(tokenize_char(text), text_cfg.max_len)
    ids = vocab.encode(tokens)
    x, lengths, _ = collate_batch([(ids, 0)], pad_id=vocab.pad_id)
    x, lengths = x.to(device), lengths.to(device)
    logits = model(x, lengths)[0]
    probs = torch.softmax(logits, dim=-1)
    score = float(probs[1].item())
    pred = int(torch.argmax(probs).item())
    return pred, score


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--artifacts", type=str, required=True, help="directory containing model.pt/vocab.json/config.json")
    p.add_argument("--text", type=str, required=True)
    p.add_argument("--device", type=str, default="cpu")
    args = p.parse_args()

    device = torch.device(args.device)
    model, vocab, _, text_cfg = load_artifacts(args.artifacts, device=device)
    pred, score = predict_one(model, vocab, text_cfg, args.text, device=device)
    print({"pred": pred, "score": score})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

