# LSTM 功能包

这整个文件夹就是“规则 + LSTM 语义分析 + 服务化接口”的可运行实现。

## 最短跑通

```powershell
python preflight.py
pip install -r requirements-lstm-service.txt
pip install torch==2.7.0+cpu --index-url https://download.pytorch.org/whl/cpu
python run_train.py --epochs 10
python run_fastapi.py --host 0.0.0.0 --port 8000
```

完整说明见 `README_LSTM_SERVICE.md`。

## 双击运行（Windows）

如果你是直接在资源管理器里“双击 .py”导致窗口一闪就关，改用这两个脚本（会自动 `cd` 到当前目录并 `pause`）：

- 训练：`run_train.bat`
- 启动服务：`run_fastapi.bat`

## 可视化界面（Streamlit）

先启动 FastAPI（`run_fastapi.bat`），再启动可视化页面：

- 可视化：`run_ui.bat`

默认打开 `http://127.0.0.1:8501`，页面会调用 `http://127.0.0.1:8000/analyze` 做检测。
