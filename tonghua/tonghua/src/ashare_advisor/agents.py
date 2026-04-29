from __future__ import annotations

import pandas as pd


POSITIVE_WORDS = ["增长", "中标", "回购", "增持", "分红", "预增", "盈利", "净利润"]
NEGATIVE_WORDS = ["亏损", "下滑", "减持", "立案", "处罚", "退市", "诉讼", "冻结", "问询", "警示"]

RATING_META = {
    "buy": ("买入", "10%-20% 分批建仓，避免一次性追高", "#dc2626"),
    "trial_buy": ("增配", "5%-10% 小仓试探，等确认后再加", "#ea580c"),
    "hold": ("持有", "已有仓位继续观察，严格盯住风险线", "#2563eb"),
    "watch": ("持有", "未持有先等待，已有仓位保持轻仓观察", "#64748b"),
    "reduce": ("降配", "先降 1/3 到 1/2 仓位，剩余仓位设纪律线", "#9333ea"),
    "avoid": ("卖出", "不新开仓，已有仓位优先做风险处理", "#16a34a"),
}


def _safe_float(value, default: float | None = None) -> float | None:
    try:
        if value is None or pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def _fmt_pct(value: float | None) -> str:
    if value is None:
        return "未知"
    return f"{value * 100:.1f}%"


def _fmt_amount(value: float | None) -> str:
    if value is None:
        return "未知"
    sign = "-" if value < 0 else ""
    value = abs(float(value))
    if value >= 1e8:
        return f"{sign}{value / 1e8:.2f}亿"
    if value >= 1e4:
        return f"{sign}{value / 1e4:.1f}万"
    return f"{sign}{value:.0f}"


def _stance(score: float) -> str:
    if score >= 70:
        return "偏多"
    if score >= 58:
        return "谨慎偏多"
    if score <= 40:
        return "偏空"
    if score <= 48:
        return "谨慎偏空"
    return "中性"


def _agent(role: str, score: float, evidence: list[str], concerns: list[str]) -> dict:
    score = max(0.0, min(100.0, float(score)))
    return {
        "role": role,
        "score": round(score, 1),
        "stance": _stance(score),
        "evidence": evidence[:3] or ["暂未形成明确加分项"],
        "concerns": concerns[:3] or ["暂无突出的单项风险"],
    }


def _headline_text(df: pd.DataFrame, columns: list[str]) -> str:
    if not isinstance(df, pd.DataFrame) or df.empty:
        return ""
    cols = [col for col in columns if col in df.columns]
    if not cols:
        return ""
    values = df[cols].head(8).to_numpy().ravel()
    return " ".join(str(value) for value in values if pd.notna(value))


def _technical_agent(result: dict) -> dict:
    frame = result.get("frame")
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        return _agent("技术分析师", 50, [], ["行情样本不足，技术面置信度偏低"])

    last = frame.iloc[-1]
    close = _safe_float(last.get("close"), 0) or 0
    ma20 = _safe_float(last.get("ma20"), close) or close
    ma60 = _safe_float(last.get("ma60"), close) or close
    macd_hist = _safe_float(last.get("macd_hist"), 0) or 0
    rsi = _safe_float(last.get("rsi14"), 50) or 50
    ret20 = _safe_float(last.get("ret_20"), 0) or 0
    volume_ratio = _safe_float(last.get("volume_ratio"), 1) or 1

    score = 50
    evidence: list[str] = []
    concerns: list[str] = []

    if close > ma20:
        score += 12
        evidence.append("价格站上20日均线")
    else:
        score -= 12
        concerns.append("价格仍在20日均线下方")
    if ma20 > ma60:
        score += 12
        evidence.append("20日均线高于60日均线")
    else:
        score -= 8
        concerns.append("中期均线结构尚未转强")
    if macd_hist > 0:
        score += 8
        evidence.append("MACD动能为正")
    else:
        score -= 5
        concerns.append("MACD动能偏弱")
    if 45 <= rsi <= 68:
        score += 6
        evidence.append(f"RSI约{rsi:.1f}，未明显过热")
    elif rsi > 75:
        score -= 12
        concerns.append(f"RSI约{rsi:.1f}，短线过热")
    elif rsi < 35:
        score -= 8
        concerns.append(f"RSI约{rsi:.1f}，走势偏弱")
    if ret20 > 0.18:
        score -= 8
        concerns.append(f"近20日涨幅{_fmt_pct(ret20)}，追高性价比下降")
    elif ret20 > 0.03:
        score += 5
        evidence.append(f"近20日涨幅{_fmt_pct(ret20)}，趋势温和")
    if volume_ratio > 2.8 and ret20 < 0.08:
        score -= 5
        concerns.append("放量但趋势不够强，存在分歧")

    return _agent("技术分析师", score, evidence, concerns)


def _capital_agent(result: dict) -> dict:
    fund = result.get("fund_summary") or {}
    sum_5 = _safe_float(fund.get("sum_5"))
    sum_20 = _safe_float(fund.get("sum_20"))
    latest_ratio = _safe_float(fund.get("latest_ratio"))
    score = 50
    evidence: list[str] = []
    concerns: list[str] = []

    if sum_5 is None and sum_20 is None:
        return _agent("资金分析师", 50, [], ["资金流数据暂不可用，需要降低资金面权重"])

    if sum_5 is not None:
        if sum_5 > 0:
            score += 15
            evidence.append(f"近5日主力净流入{_fmt_amount(sum_5)}")
        elif sum_5 < 0:
            score -= 15
            concerns.append(f"近5日主力净流出{_fmt_amount(abs(sum_5))}")
    if sum_20 is not None:
        if sum_20 > 0:
            score += 8
            evidence.append(f"近20日主力净流入{_fmt_amount(sum_20)}")
        elif sum_20 < 0:
            score -= 8
            concerns.append(f"近20日主力净流出{_fmt_amount(abs(sum_20))}")
    if latest_ratio is not None:
        if latest_ratio >= 8:
            score += 5
            evidence.append(f"最新主力净占比{latest_ratio:.1f}%")
        elif latest_ratio <= -8:
            score -= 5
            concerns.append(f"最新主力净占比{latest_ratio:.1f}%")

    return _agent("资金分析师", score, evidence, concerns)


def _fundamental_agent(result: dict) -> dict:
    financial = result.get("financial") or {}
    roe = _safe_float(financial.get("roe") or financial.get("weighted_roe"))
    revenue_growth = _safe_float(financial.get("revenue_growth"))
    profit_growth = _safe_float(financial.get("profit_growth"))
    debt_ratio = _safe_float(financial.get("debt_ratio"))
    cashflow = _safe_float(financial.get("cashflow_per_share"))
    score = 50
    evidence: list[str] = []
    concerns: list[str] = []

    if not financial:
        return _agent("基本面分析师", 50, [], ["财务数据暂不可用，基本面判断需要保守"])

    if roe is not None:
        if roe >= 15:
            score += 12
            evidence.append(f"ROE约{roe:.1f}%，盈利质量较好")
        elif roe < 5:
            score -= 10
            concerns.append(f"ROE约{roe:.1f}%，盈利能力偏弱")
    if revenue_growth is not None:
        if revenue_growth >= 15:
            score += 8
            evidence.append(f"营收同比增长{revenue_growth:.1f}%")
        elif revenue_growth < -8:
            score -= 8
            concerns.append(f"营收同比下降{abs(revenue_growth):.1f}%")
    if profit_growth is not None:
        if profit_growth >= 15:
            score += 10
            evidence.append(f"净利润同比增长{profit_growth:.1f}%")
        elif profit_growth < -10:
            score -= 12
            concerns.append(f"净利润同比下降{abs(profit_growth):.1f}%")
    if debt_ratio is not None:
        if debt_ratio > 70:
            score -= 6
            concerns.append(f"资产负债率{debt_ratio:.1f}%，杠杆偏高")
        elif debt_ratio < 45:
            score += 4
            evidence.append(f"资产负债率{debt_ratio:.1f}%，财务压力较低")
    if cashflow is not None:
        if cashflow > 0:
            score += 4
            evidence.append("每股经营现金流为正")
        else:
            score -= 5
            concerns.append("每股经营现金流为负")

    return _agent("基本面分析师", score, evidence, concerns)


def _news_agent(result: dict) -> dict:
    news = result.get("news")
    notices = result.get("notices")
    text = " ".join(
        [
            _headline_text(news, ["新闻标题", "标题", "新闻内容"]),
            _headline_text(notices, ["公告标题", "标题", "公告名称"]),
        ]
    )
    score = 50
    evidence: list[str] = []
    concerns: list[str] = []

    if not text.strip():
        return _agent("新闻公告分析师", 50, [], ["近期新闻公告数据暂不可用"])

    positive_hits = [word for word in POSITIVE_WORDS if word in text]
    negative_hits = [word for word in NEGATIVE_WORDS if word in text]
    if positive_hits:
        score += min(18, 6 + 2 * len(positive_hits))
        evidence.append(f"出现正向关键词：{'、'.join(positive_hits[:4])}")
    if negative_hits:
        score -= min(24, 8 + 3 * len(negative_hits))
        concerns.append(f"出现风险关键词：{'、'.join(negative_hits[:4])}")
    if not positive_hits and not negative_hits:
        evidence.append("近期标题未识别出明显重大风险词")

    return _agent("新闻公告分析师", score, evidence, concerns)


def _risk_agent(result: dict) -> dict:
    frame = result.get("frame")
    vol20 = None
    if isinstance(frame, pd.DataFrame) and not frame.empty:
        vol20 = _safe_float(frame.iloc[-1].get("volatility_20"))
    risks = result.get("risks") or []
    risk_level = result.get("risk_level", "中")
    score = {"低": 76, "中": 58, "高": 34}.get(risk_level, 50)
    evidence: list[str] = []
    concerns: list[str] = []

    if risk_level == "低":
        evidence.append("综合风险等级为低")
    elif risk_level == "中":
        evidence.append("综合风险处于中等区间")
    else:
        concerns.append("综合风险等级为高")
    if vol20 is not None:
        if vol20 < 0.35:
            evidence.append(f"20日年化波动{_fmt_pct(vol20)}，波动相对可控")
        elif vol20 > 0.55:
            concerns.append(f"20日年化波动{_fmt_pct(vol20)}，仓位需要压低")
    if len(risks) >= 4:
        score -= 8
        concerns.append(f"已识别{len(risks)}条风险提示")

    return _agent("风险经理", score, evidence, concerns)


def _rating_from_action(action: str) -> tuple[str, str, str]:
    return RATING_META.get(action, ("持有", "等待更清晰信号", "#64748b"))


def build_agent_review(result: dict) -> dict:
    analysts = [
        _technical_agent(result),
        _capital_agent(result),
        _fundamental_agent(result),
        _news_agent(result),
        _risk_agent(result),
    ]
    bullish = sum(1 for item in analysts if item["score"] >= 62)
    bearish = sum(1 for item in analysts if item["score"] <= 45)
    neutral = len(analysts) - bullish - bearish

    rating, position_hint, color = _rating_from_action(result.get("action", "watch"))
    if bullish - bearish >= 2:
        consensus = "多方占优"
    elif bearish - bullish >= 2:
        consensus = "空方占优"
    else:
        consensus = "分歧较大"

    levels = result.get("levels") or {}
    bull_case = list(dict.fromkeys(result.get("reasons", [])[:4]))
    bear_case = list(dict.fromkeys(result.get("risks", [])[:4]))
    risk_controls = [
        f"纪律止损位：{levels.get('stop_loss', 0):.2f}" if levels.get("stop_loss") else "先确认纪律止损位",
        f"止盈观察位：{levels.get('take_profit_watch', 0):.2f}" if levels.get("take_profit_watch") else "先确认止盈观察位",
        "若数据源提示较多，降低结论置信度，不把单次结果当成最终判断",
    ]
    watch_items = [
        "是否继续站稳20日均线",
        "主力资金5日净流入是否改善",
        "近期公告是否出现减持、处罚、亏损等风险词",
    ]

    trader_action = {
        "buy": "分批买入",
        "trial_buy": "小仓试探",
        "hold": "继续持有",
        "watch": "等待确认",
        "reduce": "分批减仓",
        "avoid": "回避或退出",
    }.get(result.get("action"), "等待确认")

    entry_text = (
        f"稳健买点 {levels.get('conservative_entry'):.2f}，突破买点 {levels.get('breakout_entry'):.2f}"
        if levels.get("conservative_entry") and levels.get("breakout_entry")
        else "等待更明确买点"
    )

    manager_view = (
        f"{consensus}：{bullish} 位角色偏多，{bearish} 位角色偏空，{neutral} 位中性。"
        f"最终采用“{rating}”评级，核心仍以风险收益比和仓位纪律为先。"
    )

    return {
        "rating": rating,
        "rating_color": color,
        "position_hint": position_hint,
        "consensus": consensus,
        "bullish_count": bullish,
        "bearish_count": bearish,
        "neutral_count": neutral,
        "analysts": analysts,
        "bull_case": bull_case or ["暂未形成明确多方论据"],
        "bear_case": bear_case or ["暂无突出的空方论据"],
        "research_manager": manager_view,
        "trader_action": trader_action,
        "entry_text": entry_text,
        "risk_controls": risk_controls,
        "watch_items": watch_items,
    }


def build_agent_markdown(result: dict, review: dict) -> str:
    lines = [
        f"# {result['code']} {result['name']} 多智能体研判",
        "",
        f"- 五档评级：{review['rating']}",
        f"- 执行动作：{review['trader_action']}",
        f"- 仓位建议：{review['position_hint']}",
        f"- 观点一致性：{review['consensus']}",
        "",
        "## 角色观点",
    ]
    for item in review["analysts"]:
        lines.extend(
            [
                f"### {item['role']}：{item['stance']}（{item['score']}）",
                "支持：",
                *[f"- {text}" for text in item["evidence"]],
                "担忧：",
                *[f"- {text}" for text in item["concerns"]],
                "",
            ]
        )
    lines.extend(
        [
            "## 多空辩论",
            "多方观点：",
            *[f"- {text}" for text in review["bull_case"]],
            "",
            "空方观点：",
            *[f"- {text}" for text in review["bear_case"]],
            "",
            "## 组合经理结论",
            review["research_manager"],
            "",
            "## 交易员执行单",
            f"- 动作：{review['trader_action']}",
            f"- 入场：{review['entry_text']}",
            f"- 仓位：{review['position_hint']}",
            "",
            "## 风险控制",
            *[f"- {text}" for text in review["risk_controls"]],
            "",
            "> 仅供个人研究和复盘，不构成投资建议。",
        ]
    )
    return "\n".join(lines)
