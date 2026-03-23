from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def _ensure_numpy() -> None:
    try:
        import numpy  # noqa: F401
    except Exception:  # noqa: BLE001
        print("[deps] numpy not found, installing ...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "numpy>=1.26"])


def _ensure_torch() -> None:
    try:
        import torch  # noqa: F401
        return
    except Exception as e:  # noqa: BLE001
        print(f"[deps] torch not available: {type(e).__name__}: {e}")

    print("[deps] installing PyTorch CPU wheel from official index ...")
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "torch==2.7.0+cpu",
            "--index-url",
            "https://download.pytorch.org/whl/cpu",
        ]
    )

    import torch  # noqa: F401  # verify import after install


def main() -> int:
    p = argparse.ArgumentParser(description="One-click training for LSTM text classifier.")
    p.add_argument("--data", type=str, default="data/sample_sensitive.csv")
    p.add_argument("--out", type=str, default="artifacts")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--max_vocab", type=int, default=5000)
    p.add_argument("--max_len", type=int, default=200)
    p.add_argument("--embed_dim", type=int, default=64)
    p.add_argument("--hidden_dim", type=int, default=64)
    p.add_argument("--num_layers", type=int, default=1)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--val_ratio", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cpu")
    args = p.parse_args()

    # Help the user avoid the "system python vs venv python" trap.
    repo_root = Path(__file__).resolve().parent
    venv_py = repo_root.parent / ".venv" / "Scripts" / "python.exe"
    if venv_py.exists() and ".venv" not in sys.executable.lower():
        print(f"[hint] current python: {sys.executable}")
        print(f"[hint] recommended: {venv_py}")

    _ensure_numpy()
    _ensure_torch()

    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"data not found: {data_path}")

    cmd = [
        sys.executable,
        "-m",
        "lstm_service.train",
        "--data",
        str(data_path),
        "--out",
        args.out,
        "--epochs",
        str(args.epochs),
        "--batch",
        str(args.batch),
        "--lr",
        str(args.lr),
        "--max_vocab",
        str(args.max_vocab),
        "--max_len",
        str(args.max_len),
        "--embed_dim",
        str(args.embed_dim),
        "--hidden_dim",
        str(args.hidden_dim),
        "--num_layers",
        str(args.num_layers),
        "--dropout",
        str(args.dropout),
        "--val_ratio",
        str(args.val_ratio),
        "--seed",
        str(args.seed),
        "--device",
        args.device,
    ]

    # Ensure cwd is LSTM/ so relative paths resolve.
    os.chdir(repo_root)
    print("[train] " + " ".join(cmd))
    subprocess.check_call(cmd)
    print(f"[ok] artifacts saved to: {Path(args.out).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

