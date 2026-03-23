from __future__ import annotations

import os
from pathlib import Path

import torch
from fastapi import FastAPI
from pydantic import BaseModel, Field

from .io import load_artifacts
from .keywords import load_keywords, rule_hit
from .predict import predict_one


class AnalyzeIn(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000)


class AnalyzeOut(BaseModel):
    is_sensitive: int
    lstm_sensitive: int
    rule_hit: int
    score: float
    threshold: float


def create_app() -> FastAPI:
    artifacts_dir = Path(os.getenv("ARTIFACTS_DIR", "artifacts"))
    keywords_path = os.getenv("KEYWORDS_PATH", "")
    threshold = float(os.getenv("THRESHOLD", "0.5"))
    device = torch.device(os.getenv("DEVICE", "cpu"))

    model, vocab, _, text_cfg = load_artifacts(artifacts_dir, device=device)
    keywords = load_keywords(keywords_path) if keywords_path else []

    app = FastAPI(title="LSTM Sensitive Text Analyzer", version="1.0.0")

    @app.get("/health")
    def health() -> dict:
        return {
            "ok": True,
            "device": str(device),
            "artifacts_dir": str(artifacts_dir),
            "keywords_loaded": len(keywords),
            "threshold": threshold,
        }

    @app.post("/analyze", response_model=AnalyzeOut)
    def analyze(req: AnalyzeIn) -> AnalyzeOut:
        pred, score = predict_one(model, vocab, text_cfg, req.text, device=device)
        lstm_sensitive = int(score >= threshold)
        hit = int(rule_hit(req.text, keywords)) if keywords else 0
        is_sensitive = int(bool(hit) or bool(lstm_sensitive))
        return AnalyzeOut(
            is_sensitive=is_sensitive,
            lstm_sensitive=lstm_sensitive,
            rule_hit=hit,
            score=score,
            threshold=threshold,
        )

    return app


app = create_app()

