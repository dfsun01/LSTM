from __future__ import annotations

import torch


@torch.no_grad()
def accuracy(logits: torch.Tensor, y: torch.Tensor) -> float:
    pred = torch.argmax(logits, dim=-1)
    return float((pred == y).float().mean().item())


@torch.no_grad()
def f1_binary(logits: torch.Tensor, y: torch.Tensor, positive_label: int = 1) -> float:
    pred = torch.argmax(logits, dim=-1)
    tp = ((pred == positive_label) & (y == positive_label)).sum().item()
    fp = ((pred == positive_label) & (y != positive_label)).sum().item()
    fn = ((pred != positive_label) & (y == positive_label)).sum().item()
    precision = tp / (tp + fp + 1e-12)
    recall = tp / (tp + fn + 1e-12)
    return float(2 * precision * recall / (precision + recall + 1e-12))

