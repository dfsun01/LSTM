from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import requests
import streamlit as st


@dataclass(frozen=True)
class ApiResult:
    is_sensitive: int
    lstm_sensitive: int
    rule_hit: int
    score: float
    threshold: float


def _load_keywords(path: str) -> list[str]:
    p = Path(path)
    if not p.exists():
        return []
    words: list[str] = []
    for line in p.read_text(encoding="utf-8").splitlines():
        w = line.strip()
        if not w or w.startswith("#"):
            continue
        words.append(w)
    return words


def _post_analyze(base_url: str, text: str, timeout_s: float = 10.0) -> ApiResult:
    url = base_url.rstrip("/") + "/analyze"
    r = requests.post(url, json={"text": text}, timeout=timeout_s)
    r.raise_for_status()
    data = r.json()
    return ApiResult(
        is_sensitive=int(data["is_sensitive"]),
        lstm_sensitive=int(data["lstm_sensitive"]),
        rule_hit=int(data["rule_hit"]),
        score=float(data["score"]),
        threshold=float(data["threshold"]),
    )


def main() -> None:
    st.set_page_config(page_title="新闻发布敏感检测", layout="centered")
    st.title("新闻发布敏感检测（规则 + LSTM）")

    with st.sidebar:
        st.subheader("连接配置")
        default_api = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")
        base_url = st.text_input("FastAPI 地址", value=default_api)
        timeout_s = st.number_input("超时(秒)", min_value=1, max_value=60, value=10, step=1)

        st.subheader("关键词（规则）")
        kw_path = st.text_input("关键词文件", value=os.getenv("KEYWORDS_PATH", "keywords.txt"))
        keywords = _load_keywords(kw_path)
        st.caption(f"已加载 {len(keywords)} 个词")
        if keywords:
            st.code("\n".join(keywords[:50]), language="text")
            if len(keywords) > 50:
                st.caption("仅展示前 50 个")

        st.subheader("示例")
        example_ok = "本市发布最新交通管制通告"
        example_bad = "刷单返现先垫付本金"
        if st.button("填充：正常示例"):
            st.session_state["text"] = example_ok
        if st.button("填充：敏感示例"):
            st.session_state["text"] = example_bad

    text = st.text_area("请输入待发布内容（标题/正文/摘要/评论）", height=180, key="text")

    col1, col2 = st.columns([1, 1])
    with col1:
        analyze_btn = st.button("检测", type="primary", use_container_width=True)
    with col2:
        st.button("清空", use_container_width=True, on_click=lambda: st.session_state.update({"text": ""}))

    if analyze_btn:
        if not text.strip():
            st.warning("请输入内容再检测。")
            st.stop()

        with st.spinner("调用模型中..."):
            try:
                res = _post_analyze(base_url=base_url, text=text, timeout_s=float(timeout_s))
            except requests.RequestException as e:
                st.error("调用失败：请确认 FastAPI 服务已启动（run_fastapi.bat / python run_fastapi.py）。")
                st.code(str(e))
                st.stop()

        if res.is_sensitive == 1:
            st.error("结果：敏感/违规（建议拦截或人工复核）")
        else:
            st.success("结果：正常（可发布）")

        st.subheader("判定分解")
        c1, c2, c3 = st.columns(3)
        c1.metric("规则命中 rule_hit", str(res.rule_hit))
        c2.metric("LSTM 判定 lstm_sensitive", str(res.lstm_sensitive))
        c3.metric("最终 is_sensitive", str(res.is_sensitive))

        st.subheader("概率与阈值")
        st.caption(f"score(敏感概率) = {res.score:.4f}，threshold = {res.threshold:.4f}")
        st.progress(min(max(res.score, 0.0), 1.0))

        st.subheader("原始返回（JSON）")
        st.code(
            json.dumps(
                {
                    "is_sensitive": res.is_sensitive,
                    "lstm_sensitive": res.lstm_sensitive,
                    "rule_hit": res.rule_hit,
                    "score": res.score,
                    "threshold": res.threshold,
                },
                ensure_ascii=False,
                indent=2,
            ),
            language="json",
        )


if __name__ == "__main__":
    main()

