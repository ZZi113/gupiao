from __future__ import annotations

import os

import requests


def build_rule_report(result: dict) -> str:
    financial = result.get("financial") or {}
    metrics = result.get("metrics") or {}
    levels = result["levels"]
    lines = [
        f"## {result['code']} {result['name']} 操作报告",
        "",
        f"**操作结论：{result['action_label']}**  ",
        f"综合分：{result['score']:.1f}，风险等级：{result['risk_level']}，行业：{result.get('industry', '未知')}",
        "",
        "### 核心判断",
        result["summary"],
        "",
        "### 操作计划",
        result["operation_plan"],
        "",
        "### 关键价位",
        f"- 稳健买点：{levels['conservative_entry']:.2f}",
        f"- 突破买点：{levels['breakout_entry']:.2f}",
        f"- 止损位：{levels['stop_loss']:.2f}",
        f"- 止盈观察位：{levels['take_profit_watch']:.2f}",
        "",
        "### 支持理由",
    ]
    lines.extend([f"- {item}" for item in result["reasons"]])
    lines.extend(["", "### 风险提示"])
    lines.extend([f"- {item}" for item in result["risks"]])
    lines.extend(
        [
            "",
            "### 数据摘要",
            f"- 最新财报期：{financial.get('report_date', '未知')}",
            f"- ROE：{metrics.get('ROE', '未知')}",
            f"- 营收增长：{metrics.get('营收增长', '未知')}",
            f"- 净利润增长：{metrics.get('净利润增长', '未知')}",
            f"- 5日主力净流入：{metrics.get('5日主力净流入', '未知')}",
            "",
            "> 仅供个人研究和复盘，不构成投资建议。",
        ]
    )
    return "\n".join(lines)


def generate_llm_report(
    result: dict,
    api_key: str | None = None,
    base_url: str | None = None,
    model: str | None = None,
) -> str:
    api_key = api_key or os.getenv("LLM_API_KEY")
    base_url = (base_url or os.getenv("LLM_BASE_URL") or "").rstrip("/")
    model = model or os.getenv("LLM_MODEL") or "gpt-4o-mini"
    fallback = build_rule_report(result)
    if not api_key or not base_url:
        return fallback

    prompt = {
        "code": result["code"],
        "name": result["name"],
        "action": result["action_label"],
        "score": result["score"],
        "risk_level": result["risk_level"],
        "summary": result["summary"],
        "operation_plan": result["operation_plan"],
        "reasons": result["reasons"],
        "risks": result["risks"],
        "metrics": result["metrics"],
    }
    try:
        resp = requests.post(
            f"{base_url}/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={
                "model": model,
                "messages": [
                    {
                        "role": "system",
                        "content": "你是谨慎的A股个人投资分析助手。只能基于用户给的数据做风险提示和操作计划，不承诺收益，不编造事实。",
                    },
                    {
                        "role": "user",
                        "content": f"请把以下结构化数据整理成自然、克制、可执行的中文操作报告：\n{prompt}",
                    },
                ],
                "temperature": 0.2,
            },
            timeout=40,
        )
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]
    except Exception as exc:
        return f"{fallback}\n\n> 大模型报告生成失败，已回退到规则报告：{type(exc).__name__}"
