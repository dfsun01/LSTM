from __future__ import annotations

import argparse
import random
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .config import ModelConfig, TextConfig
from .dataset import TextClsDataset, collate_batch, load_csv
from .io import save_artifacts
from .model import LSTMClassifier
from .metrics import accuracy, f1_binary
from .text import tokenize_char, truncate
from .vocab import build_vocab


def _split(samples: list, val_ratio: float, seed: int) -> tuple[list, list]:
    rng = random.Random(seed)
    idx = list(range(len(samples)))
    rng.shuffle(idx)
    n_val = int(round(len(samples) * val_ratio))
    val_idx = set(idx[:n_val])
    train, val = [], []
    for i, s in enumerate(samples):
        (val if i in val_idx else train).append(s)
    return train, val


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, required=True, help="CSV with headers: text,label")
    p.add_argument("--out", type=str, default="artifacts", help="output directory")
    p.add_argument("--max_vocab", type=int, default=5000)
    p.add_argument("--max_len", type=int, default=200)
    p.add_argument("--embed_dim", type=int, default=64)
    p.add_argument("--hidden_dim", type=int, default=64)
    p.add_argument("--num_layers", type=int, default=1)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--val_ratio", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cpu")
    args = p.parse_args()

    samples = load_csv(args.data)
    if len(samples) < 10:
        raise ValueError("Need at least 10 samples to train; please add more rows.")

    train_s, val_s = _split(samples, args.val_ratio, args.seed)
    text_cfg = TextConfig(max_len=args.max_len)

    tokenized = [truncate(tokenize_char(s.text), text_cfg.max_len) for s in train_s]
    vocab = build_vocab(tokenized, max_size=args.max_vocab)
    model_cfg = ModelConfig(
        vocab_size=len(vocab.itos),
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
    )

    device = torch.device(args.device)
    model = LSTMClassifier(model_cfg)
    model.to(device)

    train_ds = TextClsDataset(train_s, vocab, text_cfg.max_len)
    val_ds = TextClsDataset(val_s, vocab, text_cfg.max_len)
    collate = lambda b: collate_batch(b, pad_id=vocab.pad_id)  # noqa: E731
    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True, collate_fn=collate)
    val_dl = DataLoader(val_ds, batch_size=args.batch, shuffle=False, collate_fn=collate)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    best_val_f1 = -1.0
    out_dir = Path(args.out)

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for x, lengths, y in train_dl:
            x, lengths, y = x.to(device), lengths.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(x, lengths)
            loss = loss_fn(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total_loss += float(loss.item())

        model.eval()
        val_acc_sum = 0.0
        val_f1_sum = 0.0
        val_batches = 0
        with torch.no_grad():
            for x, lengths, y in val_dl:
                x, lengths, y = x.to(device), lengths.to(device), y.to(device)
                logits = model(x, lengths)
                val_acc_sum += accuracy(logits, y)
                val_f1_sum += f1_binary(logits, y)
                val_batches += 1

        val_acc = val_acc_sum / max(val_batches, 1)
        val_f1 = val_f1_sum / max(val_batches, 1)
        avg_loss = total_loss / max(len(train_dl), 1)
        print(f"epoch={epoch} loss={avg_loss:.4f} val_acc={val_acc:.4f} val_f1={val_f1:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            save_artifacts(out_dir, model, vocab, model_cfg, text_cfg)
            print(f"saved best artifacts to: {out_dir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
