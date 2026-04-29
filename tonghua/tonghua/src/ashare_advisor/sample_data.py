from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd


SAMPLE_NAMES = {
    "600519": ("贵州茅台", "白酒"),
    "000001": ("平安银行", "银行"),
    "300750": ("宁德时代", "电池"),
    "601318": ("中国平安", "保险"),
    "000858": ("五粮液", "白酒"),
    "600036": ("招商银行", "银行"),
    "002594": ("比亚迪", "汽车"),
    "300059": ("东方财富", "证券"),
}


def _rng(code: str) -> np.random.Generator:
    return np.random.default_rng(sum(ord(ch) for ch in code))


def make_sample_stock(code: str, days: int = 260) -> pd.DataFrame:
    rng = _rng(code)
    end = date.today()
    dates = pd.bdate_range(end=end, periods=days)

    base = 10 + (int(code[-3:]) % 90)
    drift = rng.uniform(-0.0002, 0.0008)
    vol = rng.uniform(0.012, 0.028)
    shocks = rng.normal(drift, vol, len(dates))
    cycle = np.sin(np.linspace(0, rng.uniform(6, 12), len(dates))) * rng.uniform(0.001, 0.004)
    close = base * np.exp(np.cumsum(shocks + cycle))
    open_ = close * (1 + rng.normal(0, 0.006, len(dates)))
    high = np.maximum(open_, close) * (1 + rng.uniform(0.002, 0.018, len(dates)))
    low = np.minimum(open_, close) * (1 - rng.uniform(0.002, 0.018, len(dates)))
    volume = rng.lognormal(mean=12.5, sigma=0.35, size=len(dates))
    turnover = rng.uniform(0.5, 5.5, len(dates))

    return pd.DataFrame(
        {
            "date": dates,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "amount": volume * close,
            "turnover": turnover,
        }
    )


def make_sample_profile(code: str) -> dict:
    rng = _rng(code)
    name, industry = SAMPLE_NAMES.get(code, (f"股票{code}", "未知行业"))
    return {
        "name": name,
        "industry": industry,
        "pe": round(float(rng.uniform(8, 55)), 2),
        "pb": round(float(rng.uniform(0.8, 8)), 2),
        "roe": round(float(rng.uniform(4, 26)), 2),
        "debt_ratio": round(float(rng.uniform(18, 72)), 2),
    }
