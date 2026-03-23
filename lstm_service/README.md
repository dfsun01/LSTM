# lstm_service

字符级中文 LSTM 文本二分类（训练 + 推理 + FastAPI 服务化）。

## 文件说明

- `train.py`：训练并导出 `artifacts/`（`model.pt` / `vocab.json` / `config.json`）
- `predict.py`：本地单句预测
- `api.py`：FastAPI 服务（`POST /analyze`）
- `keywords.py`：可选规则词命中（用于 `rule_hit`）

## 快速开始

在项目根目录（`c:\Users\jianr\Desktop\Python`）执行：

```powershell
pip install -r requirements-lstm-service.txt
python -m lstm_service.train --data data/sample_sensitive.csv --out artifacts --epochs 10
uvicorn lstm_service.api:app --host 0.0.0.0 --port 8000
```

请求示例：

```powershell
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/analyze -ContentType "application/json" -Body '{"text":"免费领取红包点击链接"}'
```

更完整说明见 `../README_LSTM_SERVICE.md`。
