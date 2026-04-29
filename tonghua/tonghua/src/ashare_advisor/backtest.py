from __future__ import annotations

import numpy as np
import pandas as pd


def run_ma_backtest(
    frame: pd.DataFrame,
    initial_cash: float = 100000.0,
    fee_rate: float = 0.0006,
    stop_loss: float = 0.08,
    take_profit: float = 0.18,
) -> dict:
    df = frame.dropna(subset=["close", "ma20", "ma60"]).copy().reset_index(drop=True)
    if len(df) < 80:
        return {
            "summary": {"样本不足": "至少需要80个交易日"},
            "equity": pd.DataFrame(),
            "trades": pd.DataFrame(),
        }

    cash = initial_cash
    shares = 0.0
    entry_price = 0.0
    trades = []
    equity_rows = []

    for idx, row in df.iterrows():
        close = float(row["close"])
        prev = df.iloc[idx - 1] if idx > 0 else row
        buy_signal = (
            shares == 0
            and close > float(row["ma20"]) > float(row["ma60"])
            and float(row.get("macd_hist", 0)) > 0
            and 40 <= float(row.get("rsi14", 50)) <= 72
        )
        sell_signal = False
        sell_reason = ""
        if shares > 0:
            pnl = close / entry_price - 1
            if close < float(row["ma20"]) and float(prev.get("close", close)) >= float(prev.get("ma20", close)):
                sell_signal = True
                sell_reason = "跌破20日线"
            elif pnl <= -stop_loss:
                sell_signal = True
                sell_reason = "触发止损"
            elif pnl >= take_profit and float(row.get("rsi14", 50)) > 72:
                sell_signal = True
                sell_reason = "止盈信号"

        if buy_signal:
            shares = (cash * (1 - fee_rate)) / close
            entry_price = close
            cash = 0.0
            trades.append({"日期": row["date"], "操作": "买入", "价格": close, "原因": "趋势突破"})
        elif sell_signal:
            cash = shares * close * (1 - fee_rate)
            trades.append({"日期": row["date"], "操作": "卖出", "价格": close, "原因": sell_reason})
            shares = 0.0
            entry_price = 0.0

        equity = cash + shares * close
        equity_rows.append({"date": row["date"], "equity": equity, "close": close})

    if shares > 0:
        row = df.iloc[-1]
        close = float(row["close"])
        cash = shares * close * (1 - fee_rate)
        trades.append({"日期": row["date"], "操作": "卖出", "价格": close, "原因": "期末平仓"})

    equity = pd.DataFrame(equity_rows)
    equity["return"] = equity["equity"].pct_change().fillna(0)
    equity["cummax"] = equity["equity"].cummax()
    equity["drawdown"] = equity["equity"] / equity["cummax"] - 1
    buy_hold = df.iloc[-1]["close"] / df.iloc[0]["close"] - 1
    strategy_return = equity.iloc[-1]["equity"] / initial_cash - 1
    max_drawdown = equity["drawdown"].min()

    trades_df = pd.DataFrame(trades)
    round_trips = _round_trip_returns(trades_df)
    win_rate = float((round_trips > 0).mean()) if len(round_trips) else 0.0
    annual_vol = equity["return"].std() * np.sqrt(252)
    annual_return = (1 + strategy_return) ** (252 / max(len(equity), 1)) - 1
    sharpe = annual_return / annual_vol if annual_vol and not np.isnan(annual_vol) else 0.0

    return {
        "summary": {
            "策略收益": strategy_return,
            "买入持有收益": buy_hold,
            "最大回撤": max_drawdown,
            "交易次数": int((trades_df["操作"] == "买入").sum()) if not trades_df.empty else 0,
            "胜率": win_rate,
            "年化波动": annual_vol,
            "简化夏普": sharpe,
        },
        "equity": equity,
        "trades": trades_df,
    }


def _round_trip_returns(trades: pd.DataFrame) -> pd.Series:
    if trades.empty:
        return pd.Series(dtype=float)
    buys = trades[trades["操作"] == "买入"].reset_index(drop=True)
    sells = trades[trades["操作"] == "卖出"].reset_index(drop=True)
    count = min(len(buys), len(sells))
    if count == 0:
        return pd.Series(dtype=float)
    return sells.loc[: count - 1, "价格"].reset_index(drop=True) / buys.loc[: count - 1, "价格"].reset_index(drop=True) - 1

