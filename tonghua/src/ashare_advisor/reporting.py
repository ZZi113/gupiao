from __future__ import annotations


def build_rule_report(result: dict) -> str:
    financial = result.get("financial") or {}
    metrics = result.get("metrics") or {}
    levels = result["levels"]
    components = result.get("score_components") or {}
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
        "### 分层评分",
    ]
    for name, item in components.items():
        weight = f"{float(item.get('weight', 0)) * 100:.0f}%" if item.get("available", True) else "未计入"
        evidence = "；".join(item.get("evidence") or ["暂无"])
        concerns = "；".join(item.get("concerns") or ["暂无"])
        lines.append(f"- {name}：{item.get('score', '未知')}（权重：{weight}）。依据：{evidence}。风险：{concerns}")
    lines.extend(
        [
            "",
            "### 支持理由",
        ]
    )
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
