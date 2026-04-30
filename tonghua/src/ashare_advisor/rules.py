from __future__ import annotations

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


def _clip_score(value: float) -> float:
    return float(np.clip(value, 0, 100))


def _score_between(value: float | None, low: float, high: float, soft_low: float, soft_high: float) -> float:
    if value is None or pd.isna(value):
        return 50.0
    value = float(value)
    if low <= value <= high:
        return 88.0
    if soft_low <= value < low:
        return 55.0 + 33.0 * (value - soft_low) / max(low - soft_low, 1e-9)
    if high < value <= soft_high:
        return 88.0 - 43.0 * (value - high) / max(soft_high - high, 1e-9)
    return 28.0


def _score_min(value: float | None, low: float, high: float) -> float:
    if value is None or pd.isna(value):
        return 50.0
    value = float(value)
    if value <= low:
        return 28.0
    if value >= high:
        return 90.0
    return 28.0 + 62.0 * (value - low) / max(high - low, 1e-9)


def _recent_slope(series: pd.Series, days: int) -> float:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if len(clean) <= days:
        return 0.0
    previous = float(clean.iloc[-days - 1])
    current = float(clean.iloc[-1])
    if previous <= 0:
        return 0.0
    return current / previous - 1


def _component(name: str, score: float, weight: float, evidence: list[str], concerns: list[str], available: bool = True) -> dict:
    return {
        "name": name,
        "score": round(_clip_score(score), 1),
        "weight": float(weight),
        "available": bool(available),
        "evidence": evidence[:3],
        "concerns": concerns[:3],
    }


def _weighted_component_score(components: dict[str, dict]) -> float:
    usable = [item for item in components.values() if item.get("available", True) and item.get("weight", 0) > 0]
    if not usable:
        return 50.0
    total_weight = sum(float(item["weight"]) for item in usable)
    if total_weight <= 0:
        return 50.0
    return sum(float(item["score"]) * float(item["weight"]) for item in usable) / total_weight


def _build_score_components(
    df: pd.DataFrame,
    profile: dict,
    close: float,
    ma20: float,
    ma60: float,
    ma120: float,
    rsi: float,
    macd_hist: float,
    volume_ratio: float,
    ret5: float,
    ret20: float,
    ret60: float,
    vol20: float,
    support: float,
    resistance: float,
    drawdown: float,
) -> dict[str, dict]:
    trend_score = 50.0
    trend_evidence: list[str] = []
    trend_concerns: list[str] = []
    ma20_slope = _recent_slope(df.get("ma20", pd.Series(dtype=float)), 5)
    ma60_slope = _recent_slope(df.get("ma60", pd.Series(dtype=float)), 10)
    if close > ma20 > ma60:
        trend_score += 22
        trend_evidence.append("价格位于20日、60日均线上方，趋势结构完整")
    elif close > ma20 and close > ma60:
        trend_score += 12
        trend_evidence.append("价格仍在主要均线上方")
    elif close < ma20 and close < ma60:
        trend_score -= 24
        trend_concerns.append("价格同时跌破20日和60日均线")
    elif close < ma20:
        trend_score -= 10
        trend_concerns.append("短期价格跌破20日均线")
    if ma60 > ma120:
        trend_score += 8
        trend_evidence.append("60日均线高于120日均线，中期方向仍偏上")
    elif close < ma120:
        trend_score -= 8
        trend_concerns.append("价格低于120日均线，中长期结构偏弱")
    if ma20_slope > 0.006 and ma60_slope >= 0:
        trend_score += 8
        trend_evidence.append("20日均线斜率向上，趋势在改善")
    elif ma20_slope < -0.006:
        trend_score -= 8
        trend_concerns.append("20日均线斜率向下，短期趋势转弱")
    if resistance > 0 and close >= resistance * 0.96:
        trend_score += 5
        trend_evidence.append("价格接近60日压力区，若放量突破可继续跟踪")
    if support > 0 and close <= support * 1.04:
        trend_score -= 5
        trend_concerns.append("价格靠近60日支撑区，破位风险需要盯紧")

    momentum_score = 0.34 * _score_between(ret20, 0.03, 0.15, -0.18, 0.28)
    momentum_score += 0.26 * _score_between(ret60, 0.02, 0.35, -0.30, 0.70)
    momentum_score += 0.22 * _score_between(rsi, 45, 68, 28, 82)
    momentum_score += 0.18 * (72.0 if macd_hist > 0 else 38.0)
    momentum_evidence: list[str] = []
    momentum_concerns: list[str] = []
    if macd_hist > 0:
        momentum_evidence.append("MACD柱为正，短线动能仍在")
    else:
        momentum_concerns.append("MACD柱为负，动能不足")
    if 0.03 <= ret20 <= 0.15:
        momentum_evidence.append(f"近20日涨幅{_fmt_pct(ret20)}，强度温和")
    elif ret20 > 0.18:
        momentum_concerns.append(f"近20日涨幅{_fmt_pct(ret20)}，短线偏热")
    elif ret20 < -0.12:
        momentum_concerns.append(f"近20日下跌{_fmt_pct(abs(ret20))}，修复信号不足")
    if 45 <= rsi <= 68:
        momentum_evidence.append(f"RSI {rsi:.1f} 处于健康区间")
    elif rsi > 75:
        momentum_concerns.append(f"RSI {rsi:.1f} 过热")
    elif rsi < 35:
        momentum_concerns.append(f"RSI {rsi:.1f} 偏弱")

    volume_risk_score = 0.38 * _score_between(volume_ratio, 0.85, 2.20, 0.25, 4.80)
    volume_risk_score += 0.34 * _score_between(vol20, 0.16, 0.46, 0.05, 0.75)
    volume_risk_score += 0.28 * _score_between(drawdown, -0.18, -0.02, -0.45, 0.03)
    volume_evidence: list[str] = []
    volume_concerns: list[str] = []
    if 0.85 <= volume_ratio <= 2.2:
        volume_evidence.append(f"量比{volume_ratio:.2f}，成交没有明显失真")
    elif volume_ratio > 2.8 and ret5 <= 0.02:
        volume_concerns.append("放量但短线涨幅有限，可能有分歧或派发")
    elif volume_ratio < 0.65:
        volume_concerns.append("量能偏弱，承接还需要确认")
    if vol20 <= 0.46:
        volume_evidence.append(f"20日年化波动{_fmt_pct(vol20)}，波动风险可控")
    else:
        volume_concerns.append(f"20日年化波动{_fmt_pct(vol20)}，仓位需要收缩")
    if drawdown < -0.20:
        volume_concerns.append("60日回撤较深，趋势修复难度更高")

    financial = profile.get("financial") or {}
    roe = _as_pct_value(financial.get("roe") or financial.get("weighted_roe"))
    revenue_growth = _as_pct_value(financial.get("revenue_growth"))
    profit_growth = _as_pct_value(financial.get("profit_growth"))
    debt_ratio = _as_pct_value(financial.get("debt_ratio"))
    cashflow = financial.get("cashflow_per_share")
    fundamental_available = bool(financial)
    fundamental_score = 50.0
    fundamental_evidence: list[str] = []
    fundamental_concerns: list[str] = []
    if fundamental_available:
        if roe is not None:
            fundamental_score += 12 if roe >= 15 else -10 if roe < 5 else 3
            (fundamental_evidence if roe >= 15 else fundamental_concerns if roe < 5 else fundamental_evidence).append(
                f"ROE {roe:.1f}%"
            )
        if revenue_growth is not None:
            fundamental_score += 8 if revenue_growth >= 15 else -8 if revenue_growth < -8 else 2
            (fundamental_evidence if revenue_growth >= 15 else fundamental_concerns if revenue_growth < -8 else fundamental_evidence).append(
                f"营收同比{revenue_growth:.1f}%"
            )
        if profit_growth is not None:
            fundamental_score += 10 if profit_growth >= 15 else -12 if profit_growth < -10 else 2
            (fundamental_evidence if profit_growth >= 15 else fundamental_concerns if profit_growth < -10 else fundamental_evidence).append(
                f"净利润同比{profit_growth:.1f}%"
            )
        if debt_ratio is not None:
            fundamental_score += 4 if debt_ratio < 45 else -7 if debt_ratio > 70 else 0
            if debt_ratio > 70:
                fundamental_concerns.append(f"资产负债率{debt_ratio:.1f}%偏高")
            elif debt_ratio < 45:
                fundamental_evidence.append(f"资产负债率{debt_ratio:.1f}%较低")
        if cashflow is not None and not pd.isna(cashflow):
            if cashflow > 0:
                fundamental_score += 4
                fundamental_evidence.append("经营现金流为正")
            else:
                fundamental_score -= 5
                fundamental_concerns.append("经营现金流为负")
    else:
        fundamental_concerns.append("财务数据暂不可用，不参与主评分")

    fund = _fund_summary(profile.get("fund_flow"))
    capital_available = fund.get("sum_5") is not None or fund.get("sum_20") is not None
    capital_score = 50.0
    capital_evidence: list[str] = []
    capital_concerns: list[str] = []
    if capital_available:
        sum_5 = fund.get("sum_5")
        sum_20 = fund.get("sum_20")
        latest_ratio = fund.get("latest_ratio")
        if sum_5 is not None:
            capital_score += 14 if sum_5 > 0 else -14 if sum_5 < 0 else 0
            (capital_evidence if sum_5 > 0 else capital_concerns if sum_5 < 0 else capital_evidence).append(
                f"5日主力净流{_fmt_amount(sum_5)}"
            )
        if sum_20 is not None:
            capital_score += 8 if sum_20 > 0 else -8 if sum_20 < 0 else 0
            (capital_evidence if sum_20 > 0 else capital_concerns if sum_20 < 0 else capital_evidence).append(
                f"20日主力净流{_fmt_amount(sum_20)}"
            )
        if latest_ratio is not None:
            capital_score += 5 if latest_ratio >= 8 else -5 if latest_ratio <= -8 else 0
    else:
        capital_concerns.append("资金流数据暂不可用，不参与主评分")

    news_titles = _headline_text(profile.get("news"), ["新闻标题"], ["新闻内容"])
    notice_titles = _headline_text(profile.get("notices"), ["公告标题", "标题", "公告名称"])
    text = f"{news_titles} {notice_titles}"
    news_available = bool(text.strip())
    news_score = 50.0
    news_evidence: list[str] = []
    news_concerns: list[str] = []
    if news_available:
        negative_hits = [word for word in NEGATIVE_WORDS if word in text]
        positive_hits = [word for word in POSITIVE_WORDS if word in text]
        if positive_hits:
            news_score += min(14, 5 + len(positive_hits) * 2)
            news_evidence.append(f"正向关键词：{'、'.join(positive_hits[:4])}")
        if negative_hits:
            news_score -= min(20, 6 + len(negative_hits) * 3)
            news_concerns.append(f"风险关键词：{'、'.join(negative_hits[:4])}")
    else:
        news_concerns.append("新闻公告暂不可用，不参与主评分")

    latest_date = pd.to_datetime(df.get("date"), errors="coerce").max() if "date" in df else pd.NaT
    today = pd.Timestamp.today().normalize()
    latest_day = latest_date.normalize() if not pd.isna(latest_date) else pd.NaT
    data_age_days = int((today - latest_day).days) if not pd.isna(latest_day) else 999
    warnings = profile.get("data_warnings") or []
    quality_score = 92.0
    quality_evidence: list[str] = []
    quality_concerns: list[str] = []
    if len(df) >= 120:
        quality_evidence.append(f"历史样本{len(df)}条")
    else:
        quality_score -= 16
        quality_concerns.append(f"历史样本只有{len(df)}条")
    if data_age_days <= 3:
        quality_evidence.append(f"最新行情日{latest_day.date() if not pd.isna(latest_day) else '未知'}")
    elif data_age_days <= 10:
        quality_score -= 14
        quality_concerns.append(f"行情日期距今天{data_age_days}天，可能受休市或接口延迟影响")
    else:
        quality_score -= 32
        quality_concerns.append(f"行情日期距今天{data_age_days}天，实时性不足")
    if warnings:
        quality_score -= min(24, len(warnings) * 5)
        quality_concerns.append(f"数据源提示{len(warnings)}条")
    if profile.get("realtime") and profile["realtime"].get("time"):
        quality_evidence.append(f"分钟线更新至{profile['realtime'].get('time')}")
    elif profile.get("data_notes"):
        quality_evidence.extend(str(item) for item in profile.get("data_notes", [])[:1])

    return {
        "趋势结构": _component("趋势结构", trend_score, 0.30, trend_evidence, trend_concerns),
        "动能强度": _component("动能强度", momentum_score, 0.22, momentum_evidence, momentum_concerns),
        "量价波动": _component("量价波动", volume_risk_score, 0.16, volume_evidence, volume_concerns),
        "基本面": _component("基本面", fundamental_score, 0.12, fundamental_evidence, fundamental_concerns, fundamental_available),
        "资金面": _component("资金面", capital_score, 0.08, capital_evidence, capital_concerns, capital_available),
        "消息面": _component("消息面", news_score, 0.04, news_evidence, news_concerns, news_available),
        "数据质量": _component("数据质量", quality_score, 0.08, quality_evidence, quality_concerns),
    }


def _component_summary(components: dict[str, dict]) -> tuple[list[str], list[str]]:
    evidence: list[str] = []
    concerns: list[str] = []
    for item in sorted(components.values(), key=lambda x: float(x["score"]), reverse=True):
        if item.get("available", True) and item["score"] >= 68 and item.get("evidence"):
            evidence.append(f"{item['name']}：{item['evidence'][0]}")
    for item in sorted(components.values(), key=lambda x: float(x["score"])):
        if item["score"] <= 48 and item.get("concerns"):
            concerns.append(f"{item['name']}：{item['concerns'][0]}")
    return evidence[:3], concerns[:3]


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
    legacy_score = score
    score_components = _build_score_components(
        df,
        profile,
        close,
        ma20,
        ma60,
        ma120,
        rsi,
        macd_hist,
        volume_ratio,
        ret5,
        ret20,
        ret60,
        vol20,
        support,
        resistance,
        drawdown,
    )
    component_score = _weighted_component_score(score_components)
    component_reasons, component_risks = _component_summary(score_components)
    for item in component_reasons:
        if item not in reasons:
            reasons.append(item)
    for item in component_risks:
        if item not in risks:
            risks.append(item)
    score = 0.30 * legacy_score + 0.70 * component_score

    data_quality = score_components.get("数据质量", {})
    data_quality_score = float(data_quality.get("score", 50))
    if data_quality_score < 50:
        score = min(score, 58)
        risks.insert(0, "数据质量不足，系统只保留观察/风控结论，不给出积极买入信号")
    elif data_quality_score < 68:
        score = min(score, 72)
        risks.insert(0, "数据质量一般，结论需要等下一次真实行情或补充数据确认")

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
        "score_components": score_components,
        "component_score": component_score,
        "legacy_score": legacy_score,
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
