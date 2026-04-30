from __future__ import annotations

from datetime import datetime

import pandas as pd


def _first(items: list[str] | tuple[str, ...] | None, default: str = "暂无") -> str:
    if not items:
        return default
    return str(items[0])


def _fmt_price(value) -> str:
    try:
        if value is None or pd.isna(value):
            return "N/A"
        return f"{float(value):.2f}"
    except Exception:
        return "N/A"


def _component(result: dict, name: str) -> dict:
    return (result.get("score_components") or {}).get(name) or {}


def build_decision_dashboard(result: dict, strategies: list[dict] | None = None) -> dict:
    levels = result.get("levels") or {}
    metrics = result.get("metrics") or {}
    profile = result.get("profile") or {}
    frame = result.get("frame")
    latest_date = ""
    if isinstance(frame, pd.DataFrame) and not frame.empty and "date" in frame:
        latest = pd.to_datetime(frame["date"], errors="coerce").max()
        latest_date = "" if pd.isna(latest) else latest.strftime("%Y-%m-%d")

    risk_alerts = list(result.get("risks") or [])[:5]
    positive_catalysts = list(result.get("reasons") or [])[:5]
    active_strategies = [item for item in (strategies or []) if item.get("matched") and item.get("tone") != "risk"]
    risk_strategies = [item for item in (strategies or []) if item.get("tone") == "risk"]

    one_sentence = (
        f"{result.get('name', result.get('code'))} 当前结论为“{result.get('action_label')}”，"
        f"综合分 {float(result.get('score', 0)):.1f}，核心依据是：{_first(positive_catalysts, _first(risk_alerts))}。"
    )
    if risk_strategies:
        one_sentence += f" 风险过滤器提示：{_first(risk_strategies[0].get('reasons'))}。"

    return {
        "meta": {
            "code": result.get("code"),
            "name": result.get("name"),
            "industry": result.get("industry"),
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "latest_trade_date": latest_date or profile.get("history_latest_date", ""),
            "source": result.get("source", ""),
        },
        "core_conclusion": {
            "one_sentence": one_sentence,
            "signal_type": result.get("action_label"),
            "decision_type": result.get("action"),
            "confidence": result.get("confidence"),
            "score": round(float(result.get("score", 0)), 1),
            "risk_level": result.get("risk_level"),
            "position_advice": {
                "no_position": _position_advice(result.get("action"), has_position=False),
                "has_position": _position_advice(result.get("action"), has_position=True),
            },
        },
        "data_perspective": {
            "trend": _component(result, "趋势结构"),
            "momentum": _component(result, "动能强度"),
            "volume_risk": _component(result, "量价波动"),
            "data_quality": _component(result, "数据质量"),
            "metrics": {
                "last_close": _fmt_price(result.get("last_close")),
                "ma20": metrics.get("MA20", "N/A"),
                "ma60": metrics.get("MA60", "N/A"),
                "rsi14": metrics.get("RSI14", "N/A"),
                "ret20": metrics.get("20日涨跌幅", "N/A"),
                "fund_5d": metrics.get("5日主力净流入", "N/A"),
            },
        },
        "intelligence": {
            "latest_news": _headline_text(result.get("news"), ["新闻标题", "标题"]),
            "risk_alerts": risk_alerts,
            "positive_catalysts": positive_catalysts,
            "earnings_outlook": _earnings_outlook(result),
            "sentiment_summary": _sentiment_summary(result, active_strategies, risk_strategies),
        },
        "battle_plan": {
            "sniper_points": {
                "ideal_buy": _fmt_price(levels.get("conservative_entry")),
                "breakout_buy": _fmt_price(levels.get("breakout_entry")),
                "stop_loss": _fmt_price(levels.get("stop_loss")),
                "take_profit": _fmt_price(levels.get("take_profit_watch")),
            },
            "position_strategy": {
                "suggested_position": _suggested_position(result.get("action")),
                "entry_plan": result.get("operation_plan", ""),
                "risk_control": _risk_control(result),
            },
            "action_checklist": _action_checklist(result, strategies or []),
        },
    }


def _position_advice(action: str | None, has_position: bool) -> str:
    if action in {"buy", "trial_buy"}:
        return "按计划小仓分批，不追高，不一次性打满。" if not has_position else "已有仓位可观察是否加到计划上限，仍需守止损线。"
    if action == "hold":
        return "未持有先等待回踩或有效突破。" if not has_position else "继续持有，跌破纪律线重新评估。"
    if action == "reduce":
        return "未持有不追买，等待风险释放。" if not has_position else "分批降低仓位，保留观察仓。"
    if action == "avoid":
        return "暂不新开仓。" if not has_position else "优先处理风险，必要时退出。"
    return "等待更清晰的触发条件。" if not has_position else "轻仓观察，严格控制风险。"


def _suggested_position(action: str | None) -> str:
    return {
        "buy": "10%-20% 分批",
        "trial_buy": "5%-10% 试探",
        "hold": "已有仓位持有",
        "watch": "空仓等待",
        "reduce": "降低 1/3-1/2",
        "avoid": "回避或退出",
    }.get(action or "", "等待")


def _risk_control(result: dict) -> str:
    levels = result.get("levels") or {}
    stop = _fmt_price(levels.get("stop_loss"))
    risks = result.get("risks") or []
    return f"纪律止损参考 {stop}；首要风险：{_first(risks)}"


def _action_checklist(result: dict, strategies: list[dict]) -> list[str]:
    levels = result.get("levels") or {}
    data_quality = (_component(result, "数据质量") or {}).get("score")
    items = [
        f"确认最新行情日期和数据源：{(result.get('profile') or {}).get('history_latest_date', '未知')}",
        f"只在计划价位附近行动：稳健买点 {_fmt_price(levels.get('conservative_entry'))} / 突破买点 {_fmt_price(levels.get('breakout_entry'))}",
        f"先写好退出条件：止损 {_fmt_price(levels.get('stop_loss'))}",
    ]
    if data_quality is not None and float(data_quality) < 68:
        items.insert(0, "数据质量未达高置信区间，禁止把单次结论当成最终指令")
    matched = [item["display_name"] for item in strategies if item.get("matched") and item.get("tone") != "risk"]
    if matched:
        items.append(f"当前触发策略：{'、'.join(matched[:3])}")
    return items


def _headline_text(df: pd.DataFrame, columns: list[str]) -> str:
    if not isinstance(df, pd.DataFrame) or df.empty:
        return "暂无可用新闻"
    cols = [col for col in columns if col in df.columns]
    if not cols:
        return "暂无可用新闻"
    values = [str(value) for value in df[cols].head(3).to_numpy().ravel() if pd.notna(value)]
    return "；".join(values) if values else "暂无可用新闻"


def _earnings_outlook(result: dict) -> str:
    financial = result.get("financial") or {}
    roe = financial.get("roe") or financial.get("weighted_roe")
    revenue_growth = financial.get("revenue_growth")
    profit_growth = financial.get("profit_growth")
    parts = []
    if isinstance(roe, (int, float)):
        parts.append(f"ROE {roe:.1f}%")
    if isinstance(revenue_growth, (int, float)):
        parts.append(f"营收同比 {revenue_growth:.1f}%")
    if isinstance(profit_growth, (int, float)):
        parts.append(f"净利润同比 {profit_growth:.1f}%")
    return "；".join(parts) if parts else "财务数据暂不可用，基本面权重已降低。"


def _sentiment_summary(result: dict, active: list[dict], risk: list[dict]) -> str:
    if risk:
        return f"策略风险过滤器触发：{risk[0]['display_name']}。"
    if active:
        return f"策略层较积极：{active[0]['display_name']}。"
    return f"当前以“{result.get('action_label')}”处理，等待更多确认信号。"
