from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class StrategySkill:
    name: str
    display_name: str
    category: str
    description: str


STRATEGY_LIBRARY: tuple[StrategySkill, ...] = (
    StrategySkill(
        "bull_trend",
        "多头趋势",
        "trend",
        "识别 MA5/MA10/MA20 多头排列、趋势延续和不追高的回踩机会。",
    ),
    StrategySkill(
        "ma_golden_cross",
        "均线金叉",
        "trend",
        "检查 MA5 上穿 MA10、MA10 上穿 MA20，并结合量能确认。",
    ),
    StrategySkill(
        "volume_breakout",
        "放量突破",
        "momentum",
        "识别价格突破近阶段压力位且成交量显著放大的信号。",
    ),
    StrategySkill(
        "shrink_pullback",
        "缩量回踩",
        "pullback",
        "寻找上升趋势中的缩量回踩 MA10/MA20 支撑机会。",
    ),
    StrategySkill(
        "box_oscillation",
        "箱体震荡",
        "range",
        "识别支撑压力清晰、适合等待边界确认的横盘结构。",
    ),
    StrategySkill(
        "risk_guard",
        "风险排查",
        "risk",
        "把趋势破位、过热、负面消息和数据质量不足作为交易前置过滤器。",
    ),
)


def _last_float(frame: pd.DataFrame, column: str, default: float = 0.0) -> float:
    if column not in frame or frame.empty:
        return default
    value = frame[column].iloc[-1]
    if pd.isna(value):
        return default
    return float(value)


def _crossed_above(frame: pd.DataFrame, fast: str, slow: str, lookback: int = 3) -> bool:
    if fast not in frame or slow not in frame or len(frame.dropna(subset=[fast, slow])) < lookback + 2:
        return False
    recent = frame.dropna(subset=[fast, slow]).tail(lookback + 1)
    previous = recent.iloc[:-1]
    current = recent.iloc[1:]
    return bool(((previous[fast].to_numpy() <= previous[slow].to_numpy()) & (current[fast].to_numpy() > current[slow].to_numpy())).any())


def _near(value: float, target: float, pct: float) -> bool:
    if target <= 0:
        return False
    return abs(value / target - 1) <= pct


def evaluate_strategy_skills(result: dict) -> list[dict]:
    frame = result.get("frame")
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        return []
    frame = frame.dropna(subset=["close"]).copy()
    if frame.empty:
        return []

    close = _last_float(frame, "close")
    ma5 = _last_float(frame, "ma5", close)
    ma10 = _last_float(frame, "ma10", ma5)
    ma20 = _last_float(frame, "ma20", ma10)
    ma60 = _last_float(frame, "ma60", ma20)
    volume_ratio = _last_float(frame, "volume_ratio", 1.0)
    rsi = _last_float(frame, "rsi14", 50.0)
    ret20 = _last_float(frame, "ret_20", 0.0)
    support = _last_float(frame, "support_60", float(frame["low"].tail(60).min()))
    resistance = _last_float(frame, "resistance_60", float(frame["high"].tail(60).max()))
    components = result.get("score_components") or {}
    data_quality = float((components.get("数据质量") or {}).get("score") or 50)

    rows: list[dict] = []

    bull_score = 40
    bull_reasons: list[str] = []
    if close > ma5 > ma10 > ma20:
        bull_score += 32
        bull_reasons.append("MA5 > MA10 > MA20，短中期多头排列")
    elif close > ma20 > ma60:
        bull_score += 18
        bull_reasons.append("价格站上 MA20/MA60，趋势尚未破坏")
    if ma20 > ma60:
        bull_score += 10
        bull_reasons.append("MA20 高于 MA60，中期结构偏强")
    if ret20 > 0.18 or rsi > 75:
        bull_score -= 16
        bull_reasons.append("短线涨幅或 RSI 偏热，降低追买优先级")
    rows.append(_skill_row("bull_trend", bull_score, bull_reasons))

    golden_score = 35
    golden_reasons: list[str] = []
    if _crossed_above(frame, "ma5", "ma10"):
        golden_score += 26
        golden_reasons.append("近 3 日出现 MA5 上穿 MA10")
    if _crossed_above(frame, "ma10", "ma20"):
        golden_score += 24
        golden_reasons.append("近 3 日出现 MA10 上穿 MA20")
    if volume_ratio > 1.2:
        golden_score += 8
        golden_reasons.append("金叉附近有量能确认")
    rows.append(_skill_row("ma_golden_cross", golden_score, golden_reasons))

    breakout_score = 36
    breakout_reasons: list[str] = []
    if resistance > 0 and close >= resistance * 0.985:
        breakout_score += 24
        breakout_reasons.append("价格接近或突破 60 日压力区")
    if volume_ratio >= 1.8:
        breakout_score += 22
        breakout_reasons.append(f"量比 {volume_ratio:.2f}，突破有成交量配合")
    if ret20 > 0.22 or rsi > 78:
        breakout_score -= 14
        breakout_reasons.append("突破位置偏热，需警惕假突破")
    rows.append(_skill_row("volume_breakout", breakout_score, breakout_reasons))

    pullback_score = 38
    pullback_reasons: list[str] = []
    if close > ma20 > ma60 and (_near(close, ma10, 0.035) or _near(close, ma20, 0.04)):
        pullback_score += 32
        pullback_reasons.append("上升趋势中回踩 MA10/MA20 附近")
    if volume_ratio <= 0.9:
        pullback_score += 12
        pullback_reasons.append("回踩时量能收缩，抛压相对可控")
    rows.append(_skill_row("shrink_pullback", pullback_score, pullback_reasons))

    box_score = 42
    box_reasons: list[str] = []
    if resistance > support > 0:
        box_width = resistance / support - 1
        position = (close - support) / max(resistance - support, 1e-9)
        if 0.08 <= box_width <= 0.32 and 0.18 <= position <= 0.82:
            box_score += 24
            box_reasons.append("支撑压力区间较清晰，价格位于箱体内部")
        elif position < 0.18:
            box_score += 10
            box_reasons.append("靠近箱体支撑，等待企稳确认")
        elif position > 0.82:
            box_score += 10
            box_reasons.append("靠近箱体压力，等待突破确认")
    rows.append(_skill_row("box_oscillation", box_score, box_reasons))

    risk_score = 70
    risk_reasons: list[str] = []
    if close < ma20 and close < ma60:
        risk_score += 16
        risk_reasons.append("价格同时跌破 MA20/MA60")
    if ret20 > 0.22 or rsi > 78:
        risk_score += 12
        risk_reasons.append("短线过热，追高风险上升")
    if data_quality < 68:
        risk_score += 12
        risk_reasons.append("数据质量一般，需降低结论置信度")
    if result.get("risks"):
        risk_reasons.append(str(result["risks"][0]))
    rows.append(_skill_row("risk_guard", risk_score, risk_reasons, higher_is_risk=True))

    return sorted(rows, key=lambda item: item["priority"], reverse=True)


def _skill_row(name: str, score: float, reasons: list[str], higher_is_risk: bool = False) -> dict:
    skill = next(item for item in STRATEGY_LIBRARY if item.name == name)
    score = max(0.0, min(100.0, float(score)))
    if higher_is_risk:
        matched = score >= 68
        priority = score if matched else 100 - score
    else:
        matched = score >= 62
        priority = score
    return {
        "name": skill.name,
        "display_name": skill.display_name,
        "category": skill.category,
        "description": skill.description,
        "score": round(score, 1),
        "matched": matched,
        "tone": "risk" if higher_is_risk and matched else "active" if matched else "neutral",
        "priority": round(priority, 1),
        "reasons": reasons[:3] or ["暂未触发该策略的核心条件"],
    }
