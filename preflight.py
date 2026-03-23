from __future__ import annotations

import importlib
import platform
import sys


def _check_import(name: str) -> tuple[bool, str]:
    try:
        importlib.import_module(name)
        return True, "OK"
    except Exception as e:  # noqa: BLE001
        return False, f"{type(e).__name__}: {e}"


def main() -> int:
    print("python =", sys.executable)
    print("version =", sys.version.replace("\n", " "))
    print("platform =", platform.platform())
    print()

    for m in ["fastapi", "pydantic", "uvicorn", "torch"]:
        ok, msg = _check_import(m)
        print(f"{m:8s} => {msg}")

    ok_torch, msg_torch = _check_import("torch")
    if not ok_torch:
        print("\n[解决建议]")
        print("1) 先确保你用的是虚拟环境的 Python：")
        print(r"   cd LSTM")
        print(r"   ..\.venv\Scripts\Activate.ps1")
        print("2) 用官方 CPU 源安装 PyTorch（Windows 更稳）：")
        print(r"   ..\.venv\Scripts\python -m pip install torch==2.7.0+cpu --index-url https://download.pytorch.org/whl/cpu")
        print("3) 如果仍然报 WinError 1114（c10.dll 初始化失败），再考虑改用 Python 3.11 x64 重建 venv。")
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
