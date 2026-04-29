from __future__ import annotations

import math
from datetime import datetime

import numpy as np
import pandas as pd

from .data import DataProvider
from .sample_data import SAMPLE_NAMES


MODE_LABELS = {
    "balanced": "稳健优质",
    "momentum": "趋势增强",
    "value": "低估修复",
}


def _num(value, default=np.nan):
    try:
        if value in ("-", "--", "", None):
            return default
        out = pd.to_numeric(value, errors="coerce")
        return default if pd.isna(out) else float(out)
    except Exception:
        return default


def _score_between(value: float, low: float, high: float, soft_low: float, soft_high: float) -> float:
    if np.isnan(value):
        return 45.0
    if low <= value <= high:
        return 90.0
    if soft_low <= value < low:
        return 65.0 + 25.0 * (value - soft_low) / max(low - soft_low, 1e-9)
    if high < value <= soft_high:
        return 90.0 - 45.0 * (value - high) / max(soft_high - high, 1e-9)
    return 25.0


def _score_min(value: float, low: float, high: float) -> float:
    if np.isnan(value):
        return 45.0
    if value >= high:
        return 95.0
    if value <= low:
        return 25.0
    return 25.0 + 70.0 * (value - low) / max(high - low, 1e-9)


def _fmt_money(value: float | None) -> str:
    if value is None or pd.isna(value):
        return "未知"
    value = float(value)
    if abs(value) >= 1e8:
        return f"{value / 1e8:.2f}亿"
    if abs(value) >= 1e4:
        return f"{value / 1e4:.1f}万"
    return f"{value:.0f}"


def _sample_snapshot() -> pd.DataFrame:
    rows = []
    for idx, (code, (name, _industry)) in enumerate(SAMPLE_NAMES.items(), start=1):
        base = 8 + int(code[-2:])
        rows.append(
            {
                "代码": code,
                "名称": name,
                "最新价": float(base),
                "涨跌幅": (idx % 5 - 2) * 1.2,
                "成交额": (idx + 2) * 1.2e8,
                "换手率": 1.0 + idx * 0.35,
                "市盈率-动态": 8 + idx * 5,
                "市净率": 0.9 + idx * 0.55,
                "总市值": (idx + 5) * 1.5e10,
                "流通市值": (idx + 4) * 1.1e10,
                "量比": 0.8 + idx * 0.12,
                "5分钟涨跌": (idx % 3 - 1) * 0.4,
                "60日涨跌幅": (idx % 6 - 2) * 8.0,
                "年初至今涨跌幅": (idx % 7 - 3) * 6.0,
            }
        )
    return pd.DataFrame(rows)


def load_market_snapshot() -> tuple[pd.DataFrame, str, list[str]]:
    provider = DataProvider()
    warnings: list[str] = []
    if provider.ak is None:
        return _sample_snapshot(), "演示全市场快照", ["未安装 AKShare，当前使用演示候选池。"]
    try:
        df = provider.ak.stock_zh_a_spot_em()
        if isinstance(df, pd.DataFrame) and not df.empty:
            return df.copy(), f"东方财富全A快照 {datetime.now():%H:%M:%S}", warnings
    except Exception as exc:
        warnings.append(f"全市场快照接口不可用：{type(exc).__name__}")
    return _sample_snapshot(), "演示全市场快照", warnings


def _normalise_snapshot(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()
    required = [
        "代码",
        "名称",
        "最新价",
        "涨跌幅",
        "成交额",
        "换手率",
        "市盈率-动态",
        "市净率",
        "总市值",
        "流通市值",
        "量比",
        "5分钟涨跌",
        "60日涨跌幅",
        "年初至今涨跌幅",
    ]
    for col in required:
        if col not in df:
            df[col] = np.nan if col not in {"代码", "名称"} else ""
    df["代码"] = df["代码"].astype(str).str.extract(r"(\d{6})", expand=False).fillna("")
    df["名称"] = df["名称"].astype(str)
    for col in required:
        if col not in {"代码", "名称"}:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df[required].copy()


def _base_filter(df: pd.DataFrame, mode: str) -> pd.DataFrame:
    out = df.copy()
    name = out["名称"].fillna("")
    mask = (
        out["代码"].str.match(r"^[036]\d{5}$")
        & ~name.str.contains("ST|退|退市", regex=True, case=False, na=False)
        & ~name.str.startswith(("N", "C"))
        & out["最新价"].between(2, 5000)
        & out["成交额"].fillna(0).ge(5e7)
        & out["总市值"].fillna(0).ge(3e9)
        & out["市盈率-动态"].fillna(-1).gt(0)
        & out["市净率"].fillna(-1).gt(0)
        & out["换手率"].fillna(0).between(0.2, 20)
        & out["涨跌幅"].fillna(0).between(-8, 8)
    )
    if mode == "balanced":
        mask &= out["市盈率-动态"].between(4, 60) & out["市净率"].between(0.4, 9)
    elif mode == "momentum":
        mask &= out["60日涨跌幅"].fillna(0).between(3, 90) & out["涨跌幅"].fillna(0).between(-3, 7.5)
    elif mode == "value":
        mask &= out["市盈率-动态"].between(3, 35) & out["市净率"].between(0.3, 4.5)
    return out[mask].copy()


def _quick_score(row: pd.Series, mode: str) -> tuple[float, list[str]]:
    pe = _num(row.get("市盈率-动态"))
    pb = _num(row.get("市净率"))
    amount = _num(row.get("成交额"))
    market_cap = _num(row.get("总市值"))
    turnover = _num(row.get("换手率"))
    change = _num(row.get("涨跌幅"), 0)
    ret60 = _num(row.get("60日涨跌幅"), 0)

    valuation = 0.55 * _score_between(pe, 8, 35, 3, 80) + 0.45 * _score_between(pb, 0.7, 5.5, 0.2, 12)
    liquidity = 0.55 * _score_min(math.log10(max(amount, 1)), 7.6, 9.6) + 0.45 * _score_min(
        math.log10(max(market_cap, 1)), 9.8, 11.6
    )
    momentum = 0.65 * _score_between(ret60, 3, 45, -30, 95) + 0.35 * _score_between(change, -1.5, 4.5, -8, 8)
    activity = _score_between(turnover, 0.8, 8, 0.1, 20)

    if mode == "momentum":
        score = 0.44 * momentum + 0.24 * liquidity + 0.20 * valuation + 0.12 * activity
    elif mode == "value":
        score = 0.46 * valuation + 0.24 * liquidity + 0.18 * momentum + 0.12 * activity
    else:
        score = 0.34 * valuation + 0.28 * liquidity + 0.26 * momentum + 0.12 * activity

    reasons: list[str] = []
    if pe and pe > 0:
        reasons.append(f"PE约{pe:.1f}")
    if pb and pb > 0:
        reasons.append(f"PB约{pb:.1f}")
    if amount and amount > 0:
        reasons.append(f"成交额{_fmt_money(amount)}")
    if ret60 is not None and not np.isnan(ret60):
        reasons.append(f"60日涨跌幅{ret60:.1f}%")
    return float(np.clip(score, 0, 100)), reasons


def screen_market_candidates(raw: pd.DataFrame, mode: str = "balanced", limit: int = 60) -> pd.DataFrame:
    df = _normalise_snapshot(raw)
    df = _base_filter(df, mode)
    if df.empty:
        return df
    scored = []
    for _, row in df.iterrows():
        score, reasons = _quick_score(row, mode)
        item = row.to_dict()
        item["初筛分"] = round(score, 1)
        item["初筛理由"] = "；".join(reasons[:4])
        scored.append(item)
    return pd.DataFrame(scored).sort_values("初筛分", ascending=False).head(limit).reset_index(drop=True)
