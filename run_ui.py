from __future__ import annotations

import argparse
import subprocess
import sys


def _ensure_deps() -> None:
    try:
        import streamlit  # noqa: F401
        import requests  # noqa: F401
    except Exception:  # noqa: BLE001
        print("[deps] streamlit/requests not found, installing ...")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "streamlit>=1.32", "requests>=2.31"]
        )


def main() -> int:
    p = argparse.ArgumentParser(description="Quick start Streamlit UI.")
    p.add_argument("--port", type=int, default=8501)
    p.add_argument("--host", type=str, default="127.0.0.1")
    args, unknown = p.parse_known_args()

    _ensure_deps()

    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        "ui_streamlit.py",
        "--server.address",
        args.host,
        "--server.port",
        str(args.port),
    ] + unknown
    raise SystemExit(subprocess.call(cmd))


if __name__ == "__main__":
    raise SystemExit(main())

