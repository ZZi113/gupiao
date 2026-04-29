from __future__ import annotations

import math
import time
from datetime import datetime

import numpy as np
import pandas as pd
import requests

from .data import DataProvider
from .sample_data import SAMPLE_NAMES


MODE_LABELS = {
    "balanced": "稳健优质",
    "momentum": "趋势增强",
    "value": "低估修复",
    "quality_growth": "成长质量",
    "capital_inflow": "资金关注",
    "breakout": "突破跟踪",
    "oversold_rebound": "超跌修复",
}

MODE_DESCRIPTIONS = {
    "balanced": "估值不过热、成交活跃、趋势不弱，适合做普通候选池。",
    "momentum": "更重视60日趋势和当日强度，适合找强势延续机会。",
    "value": "更重视PE/PB处于相对合理区间，适合找低估修复候选。",
    "quality_growth": "先用估值和流动性控制风险，再交给深度复核看ROE、营收和利润增长。",
    "capital_inflow": "更重视成交额、换手和量比，深度复核时再确认主力资金流。",
    "breakout": "偏向趋势刚走强或准备突破的股票，后续要看突破买点和止损位。",
    "oversold_rebound": "寻找阶段跌幅较大但仍有流动性的修复候选，风险相对更高。",
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


def _request_json(url: str, params: dict, timeout: int = 14) -> dict:
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/122.0 Safari/537.36",
        "Referer": "https://quote.eastmoney.com/center/gridlist.html#hs_a_board",
    }
    last_error: Exception | None = None
    for trust_env in (True, False):
        try:
            session = requests.Session()
            session.trust_env = trust_env
            response = session.get(url, params=params, headers=headers, timeout=timeout)
            response.raise_for_status()
            return response.json()
        except Exception as exc:
            last_error = exc
    raise last_error or RuntimeError("request failed")


def _load_spot_eastmoney_direct() -> pd.DataFrame:
    url_candidates = [
        "https://82.push2.eastmoney.com/api/qt/clist/get",
        "https://push2.eastmoney.com/api/qt/clist/get",
    ]
    fields = (
        "f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f12,f13,f14,f15,f16,f17,f18,"
        "f20,f21,f23,f24,f25,f22,f11,f62,f128,f136,f115,f152"
    )
    base_params = {
        "pn": "1",
        "pz": "500",
        "po": "1",
        "np": "1",
        "ut": "bd1d9ddb04089700cf9c27f6f7426281",
        "fltt": "2",
        "invt": "2",
        "fid": "f12",
        "fs": "m:0 t:6,m:0 t:80,m:1 t:2,m:1 t:23,m:0 t:81 s:2048",
        "fields": fields,
        "_": int(time.time() * 1000),
    }
    board_fs = [
        "m:0 t:6",
        "m:0 t:80",
        "m:1 t:2",
        "m:1 t:23",
        "m:0 t:81 s:2048",
    ]

    def fetch_rows(url: str, fs: str) -> list[dict]:
        params = dict(base_params)
        params["fs"] = fs
        first = _request_json(url, params)
        data = first.get("data") or {}
        total = int(data.get("total") or 0)
        page_size = int(params["pz"])
        total_pages = max(1, math.ceil(total / page_size))
        page_rows = list(data.get("diff") or [])
        for page in range(2, total_pages + 1):
            page_params = dict(params)
            page_params["pn"] = str(page)
            page_params["_"] = int(time.time() * 1000)
            page_data = (_request_json(url, page_params).get("data") or {}).get("diff") or []
            page_rows.extend(page_data)
        return page_rows

    rows: list[dict] = []
    last_error: Exception | None = None
    for url in url_candidates:
        try:
            rows = fetch_rows(url, str(base_params["fs"]))
            if len(rows) < 2000:
                board_rows: list[dict] = []
                for fs in board_fs:
                    board_rows.extend(fetch_rows(url, fs))
                if len(board_rows) > len(rows):
                    rows = board_rows
            break
        except Exception as exc:
            rows = []
            last_error = exc
            continue
    if not rows:
        raise last_error or RuntimeError("empty eastmoney spot")
    mapped = []
    seen: set[str] = set()
    for idx, row in enumerate(rows, start=1):
        code = str(row.get("f12") or "")
        if not code or code in seen:
            continue
        seen.add(code)
        mapped.append(
            {
                "序号": idx,
                "代码": code,
                "名称": row.get("f14"),
                "最新价": row.get("f2"),
                "涨跌幅": row.get("f3"),
                "涨跌额": row.get("f4"),
                "成交量": row.get("f5"),
                "成交额": row.get("f6"),
                "振幅": row.get("f7"),
                "换手率": row.get("f8"),
                "市盈率-动态": row.get("f9"),
                "量比": row.get("f10"),
                "5分钟涨跌": row.get("f11"),
                "最高": row.get("f15"),
                "最低": row.get("f16"),
                "今开": row.get("f17"),
                "昨收": row.get("f18"),
                "总市值": row.get("f20"),
                "流通市值": row.get("f21"),
                "涨速": row.get("f22"),
                "市净率": row.get("f23"),
                "60日涨跌幅": row.get("f24"),
                "年初至今涨跌幅": row.get("f25"),
                "主力净流入": row.get("f62"),
                "市盈率TTM": row.get("f115"),
            }
        )
    return pd.DataFrame(mapped)


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
    warnings: list[str] = []
    try:
        df = _load_spot_eastmoney_direct()
        if isinstance(df, pd.DataFrame) and not df.empty:
            return df.copy(), f"东方财富全A快照直连 {datetime.now():%H:%M:%S}", warnings
    except Exception as exc:
        warnings.append(f"东方财富全A直连不可用：{type(exc).__name__}")

    provider = DataProvider()
    if provider.ak is None:
        warnings.append("未安装 AKShare，已跳过 AKShare 备用全市场快照。")
        return _sample_snapshot(), "演示全市场快照", warnings

    try:
        df = provider.ak.stock_zh_a_spot_em()
        if isinstance(df, pd.DataFrame) and not df.empty:
            return df.copy(), f"AKShare东方财富全A快照 {datetime.now():%H:%M:%S}", warnings
    except Exception as exc:
        warnings.append(f"AKShare全市场快照不可用：{type(exc).__name__}")
    try:
        df = provider.ak.stock_zh_a_spot()
        if isinstance(df, pd.DataFrame) and not df.empty:
            return df.copy(), f"AKShare新浪全A快照 {datetime.now():%H:%M:%S}", warnings
    except Exception as exc:
        warnings.append(f"AKShare新浪全A快照不可用：{type(exc).__name__}")
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
        "主力净流入",
        "市盈率TTM",
        "振幅",
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
    pe = out["市盈率-动态"]
    pb = out["市净率"]
    turnover = out["换手率"]
    market_cap = out["总市值"]
    has_pe = pe.notna().any()
    has_pb = pb.notna().any()
    has_turnover = turnover.notna().any()
    has_market_cap = market_cap.notna().any()
    market_ok = market_cap.fillna(0).ge(2e9) if has_market_cap else pd.Series(True, index=out.index)
    pe_ok = pe.fillna(1).gt(0) if has_pe else pd.Series(True, index=out.index)
    pb_ok = pb.fillna(1).gt(0) if has_pb else pd.Series(True, index=out.index)
    turnover_ok = turnover.fillna(1).between(0.15, 25) if has_turnover else pd.Series(True, index=out.index)
    mask = (
        out["代码"].str.match(r"^[036]\d{5}$")
        & ~name.str.contains("ST|退|退市", regex=True, case=False, na=False)
        & ~name.str.startswith(("N", "C"))
        & out["最新价"].between(2, 5000)
        & out["成交额"].fillna(0).ge(3e7)
        & market_ok
        & pe_ok
        & pb_ok
        & turnover_ok
        & out["涨跌幅"].fillna(0).between(-9.5, 9.5)
    )
    if mode == "balanced":
        if has_pe:
            mask &= pe.between(4, 70) | pe.isna()
        if has_pb:
            mask &= pb.between(0.4, 10) | pb.isna()
    elif mode == "momentum":
        mask &= out["60日涨跌幅"].fillna(0).between(3, 90) & out["涨跌幅"].fillna(0).between(-3, 7.5)
    elif mode == "value":
        if has_pe:
            mask &= pe.between(3, 35)
        if has_pb:
            mask &= pb.between(0.3, 4.5)
    elif mode == "quality_growth":
        if has_pe:
            mask &= pe.between(5, 85) | pe.isna()
        if has_pb:
            mask &= pb.between(0.5, 12) | pb.isna()
    elif mode == "capital_inflow":
        mask &= out["成交额"].fillna(0).ge(1.2e8)
        if has_turnover:
            mask &= turnover.fillna(1).between(0.8, 18)
    elif mode == "breakout":
        mask &= out["60日涨跌幅"].fillna(0).between(0, 95) & out["涨跌幅"].fillna(0).between(-2.5, 8.5)
    elif mode == "oversold_rebound":
        mask &= out["60日涨跌幅"].fillna(0).between(-45, -3) & out["涨跌幅"].fillna(0).between(-6, 6)
    return out[mask].copy()


def _quick_score(row: pd.Series, mode: str) -> tuple[float, list[str]]:
    pe = _num(row.get("市盈率-动态"))
    pb = _num(row.get("市净率"))
    amount = _num(row.get("成交额"))
    market_cap = _num(row.get("总市值"))
    turnover = _num(row.get("换手率"))
    change = _num(row.get("涨跌幅"), 0)
    ret60 = _num(row.get("60日涨跌幅"), 0)
    ytd = _num(row.get("年初至今涨跌幅"), 0)
    ratio = _num(row.get("量比"), 1)
    amplitude = _num(row.get("振幅"), 0)
    main_flow = _num(row.get("主力净流入"), 0)

    valuation = 0.55 * _score_between(pe, 8, 35, 3, 80) + 0.45 * _score_between(pb, 0.7, 5.5, 0.2, 12)
    liquidity = 0.55 * _score_min(math.log10(max(amount, 1)), 7.6, 9.6) + 0.45 * _score_min(
        math.log10(max(market_cap, 1)), 9.8, 11.6
    )
    momentum = 0.65 * _score_between(ret60, 3, 45, -30, 95) + 0.35 * _score_between(change, -1.5, 4.5, -8, 8)
    activity = _score_between(turnover, 0.8, 8, 0.1, 20)
    short_term = 0.55 * _score_between(ratio, 0.8, 2.2, 0.2, 5.5) + 0.45 * _score_between(amplitude, 1.0, 7.5, 0.1, 15)
    capital = 50.0 if np.isnan(main_flow) else (82.0 if main_flow > 0 else 36.0)
    risk = 0.5 * _score_between(change, -2.5, 5.5, -9.5, 9.5) + 0.5 * _score_between(ret60, -10, 55, -55, 120)

    if mode == "momentum":
        score = 0.42 * momentum + 0.20 * liquidity + 0.18 * short_term + 0.12 * valuation + 0.08 * risk
    elif mode == "value":
        score = 0.46 * valuation + 0.22 * liquidity + 0.16 * risk + 0.10 * momentum + 0.06 * activity
    elif mode == "quality_growth":
        score = 0.28 * valuation + 0.24 * liquidity + 0.22 * momentum + 0.18 * risk + 0.08 * activity
    elif mode == "capital_inflow":
        score = 0.30 * capital + 0.28 * liquidity + 0.20 * activity + 0.14 * momentum + 0.08 * risk
    elif mode == "breakout":
        score = 0.36 * momentum + 0.24 * short_term + 0.18 * liquidity + 0.12 * capital + 0.10 * valuation
    elif mode == "oversold_rebound":
        rebound = 0.55 * _score_between(ret60, -35, -5, -70, 20) + 0.45 * _score_between(change, -1.5, 4.0, -8, 8)
        score = 0.34 * rebound + 0.24 * valuation + 0.20 * liquidity + 0.14 * risk + 0.08 * capital
    else:
        score = 0.26 * valuation + 0.24 * liquidity + 0.22 * momentum + 0.14 * risk + 0.08 * short_term + 0.06 * capital

    reasons: list[str] = []
    if pe and pe > 0:
        reasons.append(f"PE约{pe:.1f}")
    if pb and pb > 0:
        reasons.append(f"PB约{pb:.1f}")
    if amount and amount > 0:
        reasons.append(f"成交额{_fmt_money(amount)}")
    if ret60 is not None and not np.isnan(ret60):
        reasons.append(f"60日涨跌幅{ret60:.1f}%")
    if ytd is not None and not np.isnan(ytd):
        reasons.append(f"年初至今{ytd:.1f}%")
    if main_flow is not None and not np.isnan(main_flow) and main_flow != 0:
        reasons.append(f"主力净流入{_fmt_money(main_flow)}")
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
