# LSTM 文本敏感检测（可训练 + 可服务化）

## 目录

- `lstm_service/`：训练/推理/接口代码
- `data/sample_sensitive.csv`：示例训练数据（`text,label`，label=0/1）
- `keywords.txt`：可选规则词（用于接口返回 `rule_hit`）

## 0) 进入目录

在项目根目录进入 `LSTM/`：

```powershell
cd LSTM
```

## 1) 安装依赖

```powershell
python preflight.py
pip install -r requirements-lstm-service.txt
pip install torch==2.7.0+cpu --index-url https://download.pytorch.org/whl/cpu
```

## 2) 训练并导出模型

```powershell
python run_train.py --epochs 10
```

产物目录 `artifacts/`：

- `model.pt`
- `vocab.json`
- `config.json`

## 2.1) 训练数据说明（新闻管理平台发布场景）

训练脚本默认读取 `data/sample_sensitive.csv`（列：`text,label`）。

- `label=0`：正常可发布（政务公告/新闻报道/风险提示/辟谣等）
- `label=1`：敏感/违规需拦截（诈骗引流、非法交易、谣言煽动、涉黄涉赌、隐私泄露、人身攻击等）

强烈建议加入“硬负样本”（包含敏感词但语义是权威通报/辟谣/科普），避免模型变成“纯关键词命中”。

更详细的数据口径与构建建议见：`data/README.md`。

## 3) 本地预测

```powershell
python -m lstm_service.predict --artifacts artifacts --text "免费领取红包点击链接"
```

## 4) 启动 FastAPI 服务

```powershell
$env:ARTIFACTS_DIR="artifacts"
$env:KEYWORDS_PATH="keywords.txt"   # 可选
$env:THRESHOLD="0.5"               # 可选
python run_fastapi.py --host 0.0.0.0 --port 8000
```

## 4.1) 启动可视化页面（可选）

可视化页面基于 Streamlit，会调用 FastAPI 的 `/analyze` 接口展示检测结果。

```powershell
python run_ui.py
```

默认访问地址：`http://127.0.0.1:8501`

### 请求示例

```powershell
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/analyze -ContentType "application/json" -Body '{"text":"免费领取红包点击链接"}'
```

响应字段：

- `score`: LSTM 输出为“敏感(1)”的概率
- `lstm_sensitive`: `score >= threshold` 时为 1
- `rule_hit`: 关键词命中为 1（可不启用）
- `is_sensitive`: `rule_hit || lstm_sensitive` 的最终结果（便于 SpringBoot 直接用）

## 5) 返回数据字段说明（答辩/论文可直接用）

示例返回：

```json
{
  "is_sensitive": 1,
  "lstm_sensitive": 1,
  "rule_hit": 1,
  "score": 0.5424502491950989,
  "threshold": 0.5
}
```

字段含义：

- `score`：模型对“敏感(1)”这一类的预测概率（0~1，越大越倾向敏感）。
- `threshold`：判定阈值（默认 0.5，可通过环境变量 `THRESHOLD` 调整）。
- `lstm_sensitive`：LSTM 单独判定结果；当 `score >= threshold` 时取 1，否则取 0。
- `rule_hit`：规则关键词命中结果；若 `text` 命中 `keywords.txt` 中任意词则为 1，否则为 0（未配置关键词文件时恒为 0）。
- `is_sensitive`：最终综合判定结果；实现规则为 `is_sensitive = (rule_hit || lstm_sensitive)`，供 SpringBoot 直接使用。
