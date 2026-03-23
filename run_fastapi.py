from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def _ensure_deps() -> None:
    try:
        import uvicorn  # noqa: F401
        import fastapi  # noqa: F401
        import pydantic  # noqa: F401
    except Exception:  # noqa: BLE001
        print("[deps] uvicorn/fastapi/pydantic not found, installing ...")
        subprocess.check_call(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "uvicorn[standard]>=0.27",
                "fastapi>=0.110",
                "pydantic>=2.0",
            ]
        )


def main() -> int:
    p = argparse.ArgumentParser(description="Quick start FastAPI service for LSTM analyzer.")
    p.add_argument("--host", type=str, default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--reload", action="store_true")
    p.add_argument("--artifacts", type=str, default="artifacts")
    p.add_argument("--keywords", type=str, default="keywords.txt")
    p.add_argument("--threshold", type=str, default="0.5")
    args = p.parse_args()

    # Friendly hint when user accidentally uses system python.
    venv_py = Path(__file__).resolve().parent.parent / ".venv" / "Scripts" / "python.exe"
    if venv_py.exists() and "python313" in sys.executable.lower() and ".venv" not in sys.executable.lower():
        print(f"[hint] you are using system python: {sys.executable}")
        print(f"[hint] recommended: {venv_py}")

    _ensure_deps()

    artifacts_dir = Path(args.artifacts)
    if not (artifacts_dir / "model.pt").exists():
        print(f"[warn] missing artifacts in: {artifacts_dir}")
        print("       please train first:")
        print("       python -m lstm_service.train --data data/sample_sensitive.csv --out artifacts --epochs 10")

    os.environ.setdefault("ARTIFACTS_DIR", str(artifacts_dir))
    if args.keywords:
        os.environ.setdefault("KEYWORDS_PATH", str(Path(args.keywords)))
    os.environ.setdefault("THRESHOLD", str(args.threshold))

    import uvicorn

    uvicorn.run(
        "lstm_service.api:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

