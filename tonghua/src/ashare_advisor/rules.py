from __future__ import annotations

import math

import numpy as np
import pandas as pd


ACTION_META = {
    "buy": ("可买入", "#dc2626", 1),
    "trial_buy": ("小仓试探", "#ea580c", 2),
    "hold": ("继续持有", "#2563eb", 3),
    "watch": ("观察等待", "#64748b", 4),
    "reduce": ("建议减仓", "#9333ea", 5),
    "avoid": ("卖出/回避", "#16a34a", 6),
}

NEGATIVE_WORDS = [
    "亏损",
    "下滑",
    "减持",
    "立案",
    "处罚",
    "退市",
    "风险",
    "诉讼",
    "冻结",
    "问询",
    "警示",
    "*ST",
    "特别处理",
]

POSITIVE_WORDS = [
    "增长",
    "中标",
    "回购",
    "增持",
    "分红",
    "业绩预增",
    "净利润",
    "新高",
]


def _last_float(row: pd.Series, key: str, default: float = 0.0) -> float:
    value = row.get(key, default)
    if pd.isna(value) or value is None:
        return default
    return float(value)


def _period_return(df: pd.DataFrame, last: pd.Series, key: str, days: int) -> float:
    value = last.get(key)
    if value is not None and not pd.isna(value):
        return float(value)
    if len(df) <= days:
        return 0.0
    current = float(last.get("close", 0) or 0)
    previous = float(df["close"].iloc[-days - 1] or 0)
    if previous <= 0:
        return 0.0
    return current / previous - 1


def _fmt_pct(value: float | None) -> str:
    if value is None or pd.isna(value):
        return "未知"
    return f"{value * 100:.1f}%"


def _fmt_amount(value: float | None) -> str:
    if value is None or pd.isna(value):
        return "未知"
    value = float(value)
    sign = "-" if value < 0 else ""
    value = abs(value)
    if value >= 1e8:
        return f"{sign}{value / 1e8:.2f}亿"
    if value >= 1e4:
        return f"{sign}{value / 1e4:.1f}万"
    return f"{sign}{value:.0f}"


def _as_pct_value(value):
    if value is None or pd.isna(value):
        return None
    return float(value)


def _headline_text(df: pd.DataFrame, title_cols: list[str], content_cols: list[str] | None = None) -> str:
    if not isinstance(df, pd.DataFrame) or df.empty:
        return ""
    cols = [col for col in title_cols + (content_cols or []) if col in df.columns]
    if not cols:
        return ""
    return " ".join(str(v) for v in df[cols].head(8).to_numpy().ravel() if pd.notna(v))


def analyze_stock(code: str, frame: pd.DataFrame, profile: dict, holding: dict | None = None) -> dict:
    df = frame.dropna(subset=["close"]).copy()
    last = df.iloc[-1]
    close = _last_float(last, "close")
    ma20 = _last_float(last, "ma20", close)
    ma60 = _last_float(last, "ma60", close)
    ma120 = _last_float(last, "ma120", ma60)
    rsi = _last_float(last, "rsi14", 50)
    macd_hist = _last_float(last, "macd_hist", 0)
    volume_ratio = _last_float(last, "volume_ratio", 1)
    ret5 = _period_return(df, last, "ret_5", 5)
    ret10 = _period_return(df, last, "ret_10", 10)
    ret20 = _period_return(df, last, "ret_20", 20)
    ret60 = _period_return(df, last, "ret_60", 60)
    vol20 = _last_float(last, "volatility_20", 0.25)
    support = _last_float(last, "support_60", float(df["low"].tail(60).min()))
    resistance = _last_float(last, "resistance_60", float(df["high"].tail(60).max()))
    drawdown = _last_float(last, "drawdown_60", 0)

    score = 50.0
    reasons: list[str] = []
    risks: list[str] = []

    if close > ma20:
        score += 8
        reasons.append("价格站上20日均线，短期趋势没有走坏")
    else:
        score -= 10
        risks.append("价格跌破20日均线，短期趋势偏弱")

    if close > ma60 and ma20 > ma60:
        score += 12
        reasons.append("20日均线位于60日均线上方，中期结构偏强")
    elif close < ma60:
        score -= 12
        risks.append("价格低于60日均线，中期趋势需要谨慎")

    if ma60 > ma120:
        score += 5
    elif close < ma120:
        score -= 5

    if macd_hist > 0:
        score += 6
        reasons.append("MACD动能为正，短线动能仍在")
    else:
        score -= 4
        risks.append("MACD动能偏弱，暂未形成明确上行动能")

    if 1.05 <= volume_ratio <= 2.2 and ret5 > 0:
        score += 8
        reasons.append("近期上涨伴随温和放量，量价配合较好")
    elif volume_ratio > 2.8 and ret5 <= 0.02:
        score -= 10
        risks.append("出现明显放量但涨幅有限，可能存在高位分歧")
    elif volume_ratio < 0.65:
        score -= 3
        risks.append("量能偏弱，买盘承接需要继续观察")

    if 45 <= rsi <= 68:
        score += 5
        reasons.append("RSI处于相对健康区间，暂未明显过热")
    elif rsi > 75:
        score -= 12
        risks.append("RSI过热，短线追高风险较高")
    elif rsi < 35:
        score -= 8
        risks.append("RSI偏弱，仍需等待企稳信号")

    if ret20 > 0.18:
        score -= 9
        risks.append(f"近20日涨幅达到{_fmt_pct(ret20)}，短线获利盘压力较大")
    elif 0.03 <= ret20 <= 0.15:
        score += 6
        reasons.append(f"近20日涨幅{_fmt_pct(ret20)}，趋势温和")
    elif ret20 < -0.12:
        score -= 7
        risks.append(f"近20日下跌{_fmt_pct(abs(ret20))}，弱势修复仍需确认")

    if ret60 > 0.3 and drawdown > -0.05:
        score -= 8
        risks.append("阶段涨幅较大且接近高位，风险收益比下降")
    elif ret60 > 0:
        score += 4

    if vol20 > 0.55:
        score -= 8
        risks.append("20日年化波动偏高，仓位需要控制")
    elif vol20 < 0.28:
        score += 3

    score = _apply_financial_score(profile, score, reasons, risks)
    score = _apply_fund_flow_score(profile, score, reasons, risks)
    score = _apply_news_score(profile, score, reasons, risks)

    material_risk_count = len([item for item in risks if "接口" not in item and "数据" not in item])
    if material_risk_count >= 4:
        score = min(score, 68)
    elif material_risk_count >= 3:
        score = min(score, 72)

    score = float(np.clip(score, 0, 100))

    overheat = rsi > 75 or ret20 > 0.18 or (volume_ratio > 2.8 and ret5 <= 0.02)
    trend_broken = close < ma20 and close < ma60
    strong_setup = score >= 75 and not overheat and close > ma20 > ma60
    trial_setup = score >= 65 and not trend_broken and not overheat

    has_position = holding is not None and float(holding.get("cost") or 0) > 0
    cost = float(holding.get("cost") or 0) if holding else 0.0
    pnl = (close / cost - 1) if cost > 0 else None

    if trend_broken or score < 38:
        action = "avoid"
    elif has_position and (overheat or (pnl is not None and pnl > 0.18 and score < 72)):
        action = "reduce"
    elif strong_setup:
        action = "buy"
    elif trial_setup:
        action = "trial_buy"
    elif has_position and score >= 52 and not trend_broken:
        action = "hold"
    else:
        action = "watch"

    if has_position and pnl is not None:
        if pnl <= -0.08:
            action = "avoid"
            risks.insert(0, f"当前相对成本亏损{_fmt_pct(abs(pnl))}，已触发重新评估区间")
        elif pnl >= 0.15 and overheat:
            action = "reduce"
            risks.insert(0, f"当前相对成本盈利{_fmt_pct(pnl)}，且短线偏热，适合考虑锁定部分利润")

    label, color, rank = ACTION_META[action]
    conservative_entry = max(support * 1.02, ma20 * 0.985)
    breakout_entry = resistance * 1.01
    stop_loss = min(support * 0.97, ma60 * 0.98)
    if cost > 0:
        stop_loss = max(stop_loss, cost * 0.92)
    take_profit_watch = max(close * 1.12, resistance * 0.995)

    risk_level = "低" if score >= 75 and vol20 < 0.35 else "中" if score >= 50 and vol20 < 0.55 else "高"
    confidence = "高" if len(reasons) + len(risks) >= 8 else "中"

    summary = _make_summary(action, close, score, reasons, risks, pnl)
    operation_plan = _make_operation_plan(action, conservative_entry, breakout_entry, stop_loss, take_profit_watch)

    metrics = _build_metrics(profile, close, ma20, ma60, rsi, volume_ratio, ret5, ret10, ret20, ret60, vol20, pnl)
    fund_summary = _fund_summary(profile.get("fund_flow"))

    return {
        "code": code,
        "name": profile.get("name", code),
        "industry": profile.get("industry", "未知"),
        "frame": df,
        "profile": profile,
        "action": action,
        "action_label": label,
        "color": color,
        "rank": rank,
        "score": score,
        "risk_level": risk_level,
        "confidence": confidence,
        "last_close": close,
        "levels": {
            "conservative_entry": conservative_entry,
            "breakout_entry": breakout_entry,
            "stop_loss": stop_loss,
            "take_profit_watch": take_profit_watch,
        },
        "reasons": reasons[:8] or ["当前没有形成足够明确的正向信号"],
        "risks": risks[:8] or ["暂无明显单项风险，但仍需控制仓位"],
        "summary": summary,
        "operation_plan": operation_plan,
        "metrics": metrics,
        "fund_summary": fund_summary,
        "news": profile.get("news", pd.DataFrame()),
        "notices": profile.get("notices", pd.DataFrame()),
        "financial": profile.get("financial", {}),
    }


def _apply_financial_score(profile: dict, score: float, reasons: list[str], risks: list[str]) -> float:
    financial = profile.get("financial") or {}
    roe = _as_pct_value(financial.get("roe") or financial.get("weighted_roe"))
    revenue_growth = _as_pct_value(financial.get("revenue_growth"))
    profit_growth = _as_pct_value(financial.get("profit_growth"))
    debt_ratio = _as_pct_value(financial.get("debt_ratio"))
    cashflow = financial.get("cashflow_per_share")

    if roe is not None:
        if roe >= 15:
            score += 7
            reasons.append(f"最新财报ROE约{roe:.1f}%，盈利质量加分")
        elif roe < 5:
            score -= 6
            risks.append(f"最新财报ROE约{roe:.1f}%，盈利能力偏弱")
    if revenue_growth is not None:
        if revenue_growth >= 15:
            score += 5
            reasons.append(f"营收同比增长约{revenue_growth:.1f}%，基本面有增长支撑")
        elif revenue_growth < -8:
            score -= 6
            risks.append(f"营收同比下降约{abs(revenue_growth):.1f}%，基本面承压")
    if profit_growth is not None:
        if profit_growth >= 15:
            score += 6
            reasons.append(f"净利润同比增长约{profit_growth:.1f}%，业绩趋势较好")
        elif profit_growth < -10:
            score -= 8
            risks.append(f"净利润同比下降约{abs(profit_growth):.1f}%，需要警惕业绩风险")
    if debt_ratio is not None:
        if debt_ratio > 70:
            score -= 5
            risks.append(f"资产负债率约{debt_ratio:.1f}%，财务杠杆偏高")
        elif debt_ratio < 45:
            score += 3
    if cashflow is not None and not pd.isna(cashflow):
        if cashflow > 0:
            score += 3
        else:
            score -= 3
            risks.append("每股经营现金流为负，现金流质量需要确认")
    return score


def _apply_fund_flow_score(profile: dict, score: float, reasons: list[str], risks: list[str]) -> float:
    fund = profile.get("fund_flow")
    if not isinstance(fund, pd.DataFrame) or fund.empty:
        return score
    latest = fund.iloc[-1]
    main_amount = latest.get("主力净流入-净额")
    main_ratio = latest.get("主力净流入-净占比")
    recent_5 = fund.tail(5).get("主力净流入-净额", pd.Series(dtype=float)).sum()
    if pd.notna(main_amount):
        if main_amount > 0 and recent_5 > 0:
            score += 8
            reasons.append(f"主力资金近5日合计净流入{_fmt_amount(recent_5)}")
        elif main_amount < 0 and recent_5 < 0:
            score -= 8
            risks.append(f"主力资金近5日合计净流出{_fmt_amount(recent_5)}")
    if pd.notna(main_ratio):
        if main_ratio >= 8:
            score += 4
        elif main_ratio <= -8:
            score -= 4
            risks.append(f"最新主力净流入占比{main_ratio:.1f}%，资金分歧较大")
    return score


def _apply_news_score(profile: dict, score: float, reasons: list[str], risks: list[str]) -> float:
    news_titles = _headline_text(profile.get("news"), ["新闻标题"])
    notice_titles = _headline_text(profile.get("notices"), ["公告标题", "标题", "公告名称"])
    positive_text = f"{_headline_text(profile.get('news'), ['新闻标题'], ['新闻内容'])} {notice_titles}"
    negative_text = f"{news_titles} {notice_titles}"
    if not positive_text.strip() and not negative_text.strip():
        return score
    negative_hits = [word for word in NEGATIVE_WORDS if word in negative_text]
    positive_hits = [word for word in POSITIVE_WORDS if word in positive_text]
    if negative_hits:
        score -= min(12, 4 + len(negative_hits) * 2)
        risks.append(f"近期新闻/公告出现风险词：{'、'.join(negative_hits[:4])}")
    if positive_hits:
        score += min(8, 3 + len(positive_hits))
        reasons.append(f"近期新闻/公告出现正向关键词：{'、'.join(positive_hits[:4])}")
    return score


def _fund_summary(fund: pd.DataFrame | None) -> dict:
    if not isinstance(fund, pd.DataFrame) or fund.empty:
        return {"latest_main": None, "latest_ratio": None, "sum_5": None, "sum_20": None}
    return {
        "latest_main": fund.iloc[-1].get("主力净流入-净额"),
        "latest_ratio": fund.iloc[-1].get("主力净流入-净占比"),
        "sum_5": fund.tail(5).get("主力净流入-净额", pd.Series(dtype=float)).sum(),
        "sum_20": fund.tail(20).get("主力净流入-净额", pd.Series(dtype=float)).sum(),
    }


def _build_metrics(profile, close, ma20, ma60, rsi, volume_ratio, ret5, ret10, ret20, ret60, vol20, pnl):
    financial = profile.get("financial") or {}
    realtime = profile.get("realtime") or {}
    fund = _fund_summary(profile.get("fund_flow"))
    metrics = {
        "现价/收盘价": round(close, 2),
        "实时更新时间": realtime.get("time", "未获取"),
        "MA20": round(ma20, 2),
        "MA60": round(ma60, 2),
        "RSI14": round(rsi, 1),
        "量比": round(volume_ratio, 2),
        "5日涨跌幅": _fmt_pct(ret5),
        "10日涨跌幅": _fmt_pct(ret10),
        "20日涨跌幅": _fmt_pct(ret20),
        "60日涨跌幅": _fmt_pct(ret60),
        "20日年化波动": _fmt_pct(vol20),
        "最新财报期": financial.get("report_date", "未知"),
        "ROE": f"{financial.get('roe'):.1f}%" if isinstance(financial.get("roe"), (int, float)) else "未知",
        "营收增长": f"{financial.get('revenue_growth'):.1f}%" if isinstance(financial.get("revenue_growth"), (int, float)) else "未知",
        "净利润增长": f"{financial.get('profit_growth'):.1f}%" if isinstance(financial.get("profit_growth"), (int, float)) else "未知",
        "资产负债率": f"{financial.get('debt_ratio'):.1f}%" if isinstance(financial.get("debt_ratio"), (int, float)) else "未知",
        "5日主力净流入": _fmt_amount(fund.get("sum_5")),
        "20日主力净流入": _fmt_amount(fund.get("sum_20")),
    }
    if pnl is not None:
        metrics["持仓盈亏"] = _fmt_pct(pnl)
    return metrics


def _make_summary(action: str, close: float, score: float, reasons: list[str], risks: list[str], pnl: float | None) -> str:
    base = {
        "buy": "当前信号偏积极，可以纳入买入候选，但仍建议按计划分批，不追高一次性打满。",
        "trial_buy": "当前有一定机会，但信号还不够完美，更适合小仓试探或等待更舒服的位置。",
        "hold": "持仓结构暂未明显破坏，优先继续持有并盯住关键止损位。",
        "watch": "当前优势不够明显，建议观察等待，等回踩企稳或有效突破后再行动。",
        "reduce": "当前位置风险收益比转差，已有持仓可考虑分批减仓或提高止盈纪律。",
        "avoid": "当前风险信号较明显，不适合新开仓；已有持仓需要重新评估是否止损或退出。",
    }[action]
    pnl_text = f" 当前持仓盈亏约{_fmt_pct(pnl)}。" if pnl is not None else ""
    clue = reasons[0] if reasons else (risks[0] if risks else "信号中性")
    return f"{base} 现价约 {close:.2f}，综合分 {score:.1f}。核心依据：{clue}。{pnl_text}"


def _make_operation_plan(action: str, conservative_entry: float, breakout_entry: float, stop_loss: float, take_profit_watch: float) -> str:
    if action in {"buy", "trial_buy"}:
        size = "10%-20%" if action == "buy" else "5%-10%"
        return (
            f"未持有：可考虑 {size} 仓位分批，不建议高于现价太多追入；"
            f"稳健买点看 {conservative_entry:.2f} 附近，突破买点看 {breakout_entry:.2f} 附近。"
            f"若跌破 {stop_loss:.2f}，应停止加仓并重新评估。止盈观察位约 {take_profit_watch:.2f}。"
        )
    if action == "hold":
        return (
            f"已持有：继续持有为主，跌破 {stop_loss:.2f} 需要重新评估；"
            f"若上冲到 {take_profit_watch:.2f} 附近且放量滞涨，可考虑分批止盈。"
        )
    if action == "reduce":
        return (
            f"已持有：可先减掉 1/3 到 1/2 仓位，剩余仓位用 {stop_loss:.2f} 做纪律线；"
            f"未持有：不建议追买，等回落到 {conservative_entry:.2f} 附近再观察。"
        )
    if action == "avoid":
        return (
            f"未持有：暂时回避。已持有：若不能快速收回关键均线，建议降低仓位；"
            f"纪律止损位参考 {stop_loss:.2f}，不要用补仓代替止损。"
        )
    return (
        f"等待更明确的机会。稳健观察位 {conservative_entry:.2f}，有效突破位 {breakout_entry:.2f}，"
        f"风险线 {stop_loss:.2f}。"
    )


def build_market_brief(results: list[dict]) -> dict:
    buy_count = sum(1 for item in results if item["action"] in {"buy", "trial_buy"})
    watch_count = sum(1 for item in results if item["action"] in {"hold", "watch"})
    risk_count = sum(1 for item in results if item["action"] in {"reduce", "avoid"})
    avg_score = float(np.mean([item["score"] for item in results])) if results else 0.0
    return {
        "buy_count": buy_count,
        "watch_count": watch_count,
        "risk_count": risk_count,
        "avg_score": avg_score,
    }
