from __future__ import annotations

import os
import time
from datetime import date, datetime
from html import escape
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

try:
    from streamlit_autorefresh import st_autorefresh
except Exception:
    st_autorefresh = None

from .agents import build_agent_review
from .backtest import run_ma_backtest
from .data import DataProvider, normalize_codes
from .decision_dashboard import build_decision_dashboard
from .rules import analyze_stock, build_market_brief
from .screener import (
    MODE_DESCRIPTIONS,
    MODE_LABELS,
    clear_market_snapshot_file_cache,
    enrich_candidates_with_history,
    load_market_snapshot,
    screen_market_candidates,
)
from .strategy_skills import STRATEGY_LIBRARY, evaluate_strategy_skills


PAGES = ["研究", "组合", "发现", "回测", "规则"]
DISCOVERY_SCHEMA_VERSION = 6


def main() -> None:
    st.set_page_config(
        page_title="同花 A股研究台",
        page_icon="A",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    require_login()
    inject_theme()

    with st.sidebar:
        page, code_text, days, realtime, auto_refresh, refresh_seconds, holdings = render_sidebar()

    codes = normalize_codes(code_text)
    if not codes:
        render_empty_state()
        return

    labels = load_code_labels(tuple(codes))
    refresh_key = setup_auto_refresh(auto_refresh, refresh_seconds) if realtime else 0
    render_shell_header(page, codes, days, realtime, auto_refresh, refresh_seconds)

    if page == "研究":
        render_research_page(codes, labels, days, realtime, refresh_key, holdings)
    elif page == "组合":
        render_portfolio_page(codes, labels, days, realtime, refresh_key, holdings)
    elif page == "发现":
        render_discovery_page(days, realtime, refresh_key)
    elif page == "回测":
        render_backtest_page(codes, labels, days)
    else:
        render_rules_page()


def get_config_value(name: str, default: str = "") -> str:
    env_value = os.getenv(name)
    if env_value:
        return env_value
    local_secret_paths = [Path.cwd() / ".streamlit" / "secrets.toml", Path.home() / ".streamlit" / "secrets.toml"]
    if not any(path.exists() for path in local_secret_paths) and not os.getenv("STREAMLIT_SHARING_MODE"):
        return default
    try:
        value = st.secrets.get(name, default)
    except Exception:
        return default
    return str(value) if value is not None else default


def get_query_value(name: str, default: str = "") -> str:
    try:
        value = st.query_params.get(name, default)
    except Exception:
        value = default
    if isinstance(value, list):
        return str(value[0]) if value else default
    return str(value) if value is not None else default


def set_query_value(name: str, value: str) -> None:
    try:
        st.query_params[name] = value
    except Exception:
        pass


def require_login() -> None:
    expected_password = get_config_value("APP_PASSWORD")
    if not expected_password:
        return
    if st.session_state.get("authenticated"):
        return
    st.markdown("<h1 style='margin-top:18vh'>同花 A股研究台</h1>", unsafe_allow_html=True)
    st.caption("请输入访问密码。仅供个人研究，不构成投资建议。")
    password = st.text_input("访问密码", type="password")
    if password:
        if password == expected_password:
            st.session_state["authenticated"] = True
            st.rerun()
        st.error("密码不正确")
    st.stop()


def inject_theme() -> None:
    st.markdown(
        """
        <style>
        :root {
            --bg: #f3f6fb;
            --panel: #ffffff;
            --panel-2: #f8fafc;
            --ink: #0f172a;
            --muted: #64748b;
            --line: #dbe3ef;
            --line-2: #cbd5e1;
            --brand: #0891b2;
            --brand-soft: #e6f8fb;
            --red: #ef4444;
            --green: #16a34a;
            --amber: #f59e0b;
            --nav: #0b1220;
            --nav-2: #111c2e;
        }

        .stApp {
            background:
                radial-gradient(circle at top left, rgba(8, 145, 178, .12), transparent 32rem),
                linear-gradient(180deg, #edf3f9 0%, var(--bg) 42%, #eef2f7 100%);
            color: var(--ink);
        }

        header[data-testid="stHeader"],
        div[data-testid="stToolbar"],
        div[data-testid="stDecoration"],
        #MainMenu,
        footer {
            display: none !important;
        }

        .main .block-container {
            max-width: 1680px;
            padding: 18px 24px 34px;
        }

        section[data-testid="stSidebar"] {
            background: var(--nav);
            border-right: 1px solid #1f2a3d;
            min-width: 315px !important;
            width: 315px !important;
        }

        section[data-testid="stSidebar"] > div {
            padding: 18px 14px 24px;
            width: 315px !important;
        }

        section[data-testid="stSidebar"] label,
        section[data-testid="stSidebar"] p,
        section[data-testid="stSidebar"] span {
            color: #cbd5e1;
        }

        section[data-testid="stSidebar"] div[data-testid="stButton"] button {
            background: #132035;
            border: 1px solid #2b3a55;
            color: #e2e8f0;
            box-shadow: none;
            min-height: 34px;
        }

        section[data-testid="stSidebar"] div[data-testid="stButton"] button:hover {
            background: #1a2a44;
            border-color: #50627e;
        }

        section[data-testid="stSidebar"] div[data-baseweb="select"] > div,
        section[data-testid="stSidebar"] textarea,
        section[data-testid="stSidebar"] input {
            background: #101a2b !important;
            border-color: #283850 !important;
            color: #e2e8f0 !important;
        }

        section[data-testid="stSidebar"] div[data-testid="stExpander"] {
            background: #101a2b;
            border: 1px solid #24324a;
            border-radius: 10px;
        }

        section[data-testid="stSidebar"] div[data-testid="stExpander"] summary {
            color: #f8fafc !important;
            font-weight: 850;
        }

        .brand-block {
            align-items: center;
            display: flex;
            gap: 11px;
            margin-bottom: 18px;
        }

        .brand-mark {
            align-items: center;
            background: linear-gradient(135deg, #06b6d4, #ef4444);
            border-radius: 13px;
            color: white;
            display: flex;
            font-weight: 950;
            height: 42px;
            justify-content: center;
            width: 42px;
        }

        .brand-title {
            color: #f8fafc;
            font-size: 17px;
            font-weight: 900;
            line-height: 1.1;
        }

        .brand-sub {
            color: #8da0bb;
            font-size: 10px;
            font-weight: 850;
            letter-spacing: .14em;
            margin-top: 3px;
        }

        .side-section {
            background: #101a2b;
            border: 1px solid #24324a;
            border-radius: 11px;
            margin: 0 0 12px;
            padding: 11px;
        }

        .side-kicker {
            color: #67e8f9;
            font-size: 10px;
            font-weight: 900;
            letter-spacing: .12em;
            margin-bottom: 5px;
        }

        .side-title {
            color: #f8fafc;
            font-size: 14px;
            font-weight: 850;
        }

        .terminal-header {
            align-items: center;
            background: #0b1220;
            border: 1px solid #1f2a3d;
            border-radius: 16px;
            box-shadow: 0 18px 42px rgba(15, 23, 42, .14);
            color: #e2e8f0;
            display: flex;
            gap: 14px;
            justify-content: space-between;
            margin-bottom: 12px;
            padding: 14px 16px;
        }

        .terminal-title {
            color: #f8fafc;
            font-size: 22px;
            font-weight: 950;
            line-height: 1.1;
        }

        .terminal-sub {
            color: #8da0bb;
            font-size: 12px;
            margin-top: 5px;
        }

        .status-strip {
            display: flex;
            flex-wrap: wrap;
            gap: 7px;
            justify-content: flex-end;
        }

        .status-pill {
            background: #132035;
            border: 1px solid #2b3a55;
            border-radius: 999px;
            color: #dbe7f5;
            font-size: 11px;
            font-weight: 850;
            padding: 6px 9px;
            white-space: nowrap;
        }

        .dsa-card {
            background: rgba(255, 255, 255, .96);
            border: 1px solid var(--line);
            border-radius: 14px;
            box-shadow: 0 12px 28px rgba(15, 23, 42, .06);
            padding: 14px;
        }

        .dsa-card-flat {
            background: var(--panel);
            border: 1px solid var(--line);
            border-radius: 12px;
            padding: 12px;
        }

        .label-uppercase {
            color: var(--muted);
            font-size: 10px;
            font-weight: 900;
            letter-spacing: .14em;
            text-transform: uppercase;
        }

        .hero-grid {
            display: grid;
            gap: 12px;
            grid-template-columns: minmax(0, 1.7fr) minmax(360px, .9fr);
            margin-bottom: 12px;
        }

        .decision-hero {
            background:
                linear-gradient(135deg, rgba(8, 145, 178, .14), rgba(239, 68, 68, .07)),
                #ffffff;
            border: 1px solid rgba(8, 145, 178, .22);
            border-radius: 16px;
            padding: 16px;
        }

        .decision-top {
            align-items: flex-start;
            display: flex;
            gap: 16px;
            justify-content: space-between;
        }

        .stock-name {
            color: var(--ink);
            font-size: 25px;
            font-weight: 950;
            line-height: 1.1;
        }

        .stock-meta {
            color: var(--muted);
            font-size: 12px;
            margin-top: 6px;
        }

        .score-ring {
            align-items: center;
            background: conic-gradient(var(--brand) calc(var(--score) * 1%), #e2e8f0 0);
            border-radius: 999px;
            display: flex;
            height: 94px;
            justify-content: center;
            width: 94px;
        }

        .score-ring-inner {
            align-items: center;
            background: #ffffff;
            border-radius: 999px;
            display: flex;
            flex-direction: column;
            height: 74px;
            justify-content: center;
            width: 74px;
        }

        .score-ring-inner b {
            color: var(--ink);
            font-size: 24px;
            line-height: 1;
        }

        .score-ring-inner span {
            color: var(--muted);
            font-size: 10px;
            font-weight: 900;
        }

        .one-sentence {
            color: #1e293b;
            font-size: 15px;
            line-height: 1.75;
            margin-top: 14px;
        }

        .sniper-grid {
            display: grid;
            gap: 8px;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            margin-top: 14px;
        }

        .sniper-box {
            background: #f8fafc;
            border: 1px solid #dde5ef;
            border-radius: 10px;
            padding: 9px;
        }

        .sniper-box span {
            color: var(--muted);
            display: block;
            font-size: 11px;
            margin-bottom: 3px;
        }

        .sniper-box b {
            color: var(--ink);
            font-size: 18px;
            line-height: 1.1;
        }

        .strategy-list {
            display: grid;
            gap: 8px;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            margin-top: 10px;
        }

        .strategy-item {
            background: #f8fafc;
            border: 1px solid #dde5ef;
            border-left: 4px solid #94a3b8;
            border-radius: 10px;
            padding: 9px 10px;
        }

        .strategy-item.active { border-left-color: var(--brand); }
        .strategy-item.risk { border-left-color: var(--red); }

        .strategy-head {
            align-items: center;
            display: flex;
            justify-content: space-between;
            gap: 8px;
        }

        .strategy-head b {
            color: var(--ink);
            font-size: 13px;
        }

        .strategy-score {
            color: var(--muted);
            font-size: 12px;
            font-weight: 850;
        }

        .strategy-reason {
            color: #475569;
            font-size: 12px;
            line-height: 1.55;
            margin-top: 5px;
        }

        .two-col {
            display: grid;
            gap: 12px;
            grid-template-columns: minmax(0, 1fr) minmax(0, 1fr);
        }

        .report-block {
            color: #334155;
            font-size: 14px;
            line-height: 1.75;
        }

        div[data-testid="stMetric"] {
            background: #ffffff;
            border: 1px solid var(--line);
            border-radius: 12px;
            box-shadow: none;
            padding: 10px 12px;
        }

        div[data-testid="stMetricValue"] {
            color: var(--ink);
            font-size: 23px;
        }

        div[data-testid="stMetricLabel"] p {
            color: var(--muted);
            font-size: 12px;
        }

        div[data-testid="stDataFrame"] {
            border: 1px solid var(--line);
            border-radius: 12px;
            overflow: hidden;
        }

        div[data-testid="stButton"] button,
        div[data-testid="stDownloadButton"] button {
            border-radius: 10px;
            font-weight: 800;
        }

        @media (max-width: 1100px) {
            .hero-grid,
            .two-col {
                grid-template-columns: 1fr;
            }
            .sniper-grid {
                grid-template-columns: repeat(2, minmax(0, 1fr));
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar() -> tuple[str, str, int, bool, bool, int, pd.DataFrame]:
    st.markdown(
        """
        <div class="brand-block">
            <div class="brand-mark">A</div>
            <div>
                <div class="brand-title">同花研究台</div>
                <div class="brand-sub">REAL DATA ONLY</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    requested_page = get_query_value("view", "研究")
    page = st.radio("工作区", PAGES, index=PAGES.index(requested_page) if requested_page in PAGES else 0, label_visibility="collapsed")
    set_query_value("view", page)

    default_codes = get_query_value("codes") or get_config_value("APP_DEFAULT_CODES") or "600519,000001,300750,601318,000858"
    st.markdown('<div class="side-section"><div class="side-kicker">WATCHLIST</div><div class="side-title">股票池</div></div>', unsafe_allow_html=True)
    code_text = st.text_area("股票代码", value=default_codes, height=92, help="支持逗号、空格或换行分隔。")
    if st.button("保存到当前网址", use_container_width=True):
        set_query_value("codes", ",".join(normalize_codes(code_text)))
        st.success("已保存")

    st.markdown('<div class="side-section"><div class="side-kicker">MARKET DATA</div><div class="side-title">数据窗口</div></div>', unsafe_allow_html=True)
    days = st.slider("分析周期", min_value=90, max_value=720, value=260, step=10)
    realtime = st.toggle("合并分钟线", value=True)
    auto_refresh = st.toggle("自动刷新", value=False)
    refresh_seconds = 60
    if auto_refresh:
        refresh_seconds = st.selectbox("刷新间隔", [30, 60, 120, 300], index=1, format_func=lambda x: f"{x} 秒")
    if st.button("清空缓存并重新取数", use_container_width=True):
        st.cache_data.clear()
        clear_market_snapshot_file_cache()
        st.session_state["refresh_key"] = int(st.session_state.get("refresh_key", 0)) + 1
        st.rerun()

    st.markdown('<div class="side-section"><div class="side-kicker">POSITION</div><div class="side-title">持仓成本</div></div>', unsafe_allow_html=True)
    with st.expander("录入持仓 / 成本 / 仓位", expanded=False):
        holdings = st.data_editor(
            pd.DataFrame([{"code": "", "cost": 0.0, "weight": 0.0}]),
            num_rows="dynamic",
            hide_index=True,
            use_container_width=True,
            key="terminal_holdings",
            column_config={
                "code": st.column_config.TextColumn("代码"),
                "cost": st.column_config.NumberColumn("成本", min_value=0.0, step=0.01),
                "weight": st.column_config.NumberColumn("仓位%", min_value=0.0, max_value=100.0, step=1.0),
            },
        )
    return page, code_text, days, realtime, auto_refresh, refresh_seconds, holdings


def render_empty_state() -> None:
    st.markdown(
        """
        <div class="terminal-header">
            <div>
                <div class="terminal-title">同花 A股研究台</div>
                <div class="terminal-sub">请输入至少一个 6 位 A 股代码。</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.warning("左侧股票池为空。")


def setup_auto_refresh(enabled: bool, interval_seconds: int) -> int:
    if not enabled:
        return int(st.session_state.get("refresh_key", 0))
    st.sidebar.caption(f"自动刷新：每 {interval_seconds} 秒，最近刷新 {datetime.now():%H:%M:%S}")
    if st_autorefresh is not None:
        return int(st_autorefresh(interval=interval_seconds * 1000, key="terminal_auto_refresh"))
    return int(time.time() // max(interval_seconds, 1))


def render_shell_header(page: str, codes: list[str], days: int, realtime: bool, auto_refresh: bool, refresh_seconds: int) -> None:
    realtime_text = "分钟线" if realtime else "日线"
    refresh_text = f"{refresh_seconds}s 自动" if realtime and auto_refresh else "手动"
    st.markdown(
        f"""
        <div class="terminal-header">
            <div>
                <div class="terminal-title">{escape(page)}工作台</div>
                <div class="terminal-sub">参考 daily_stock_analysis 的工作台结构：报告优先、任务清晰、真实数据失败即停止。</div>
            </div>
            <div class="status-strip">
                <span class="status-pill">WATCHLIST {len(codes)}</span>
                <span class="status-pill">WINDOW {days}D</span>
                <span class="status-pill">{escape(realtime_text)}</span>
                <span class="status-pill">{escape(refresh_text)}</span>
                <span class="status-pill">{date.today().isoformat()}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False, ttl=60 * 5)
def load_stock_cached(code: str, days: int, realtime: bool, refresh_key: int, detail_level: str) -> tuple[pd.DataFrame, dict, str]:
    provider = DataProvider()
    return provider.load_stock(code, days=days, realtime=realtime, detail_level=detail_level)


@st.cache_data(show_spinner=False, ttl=60 * 60)
def load_code_labels(codes: tuple[str, ...]) -> dict[str, str]:
    provider = DataProvider()
    return provider.load_code_name_map(codes)


def load_market_snapshot_fresh(force_refresh: bool = False) -> tuple[pd.DataFrame, str, list[str]]:
    return load_market_snapshot(force_refresh=force_refresh)


def format_stock_option(code: str, labels: dict[str, str]) -> str:
    name = labels.get(code, "")
    return f"{code} {name}" if name and name != code else code


def holding_map_from_table(holdings: pd.DataFrame) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    if not isinstance(holdings, pd.DataFrame) or holdings.empty:
        return rows
    for _, row in holdings.dropna(subset=["code"]).iterrows():
        codes = normalize_codes([row.get("code", "")])
        if codes:
            rows[codes[0]] = {"cost": float(row.get("cost") or 0), "weight": float(row.get("weight") or 0)}
    return rows


def fetch_result(code: str, labels: dict[str, str], days: int, realtime: bool, refresh_key: int, detail_level: str, holding: dict | None = None) -> dict:
    df, profile, source = load_stock_cached(code, days, realtime, refresh_key, detail_level)
    if labels.get(code):
        profile["name"] = labels[code]
    result = analyze_stock(code, df, profile, holding=holding)
    result["source"] = source
    return result


def render_research_page(codes: list[str], labels: dict[str, str], days: int, realtime: bool, refresh_key: int, holdings: pd.DataFrame) -> None:
    controls = st.columns([1.5, 1])
    selected_code = controls[0].selectbox("研究标的", codes, format_func=lambda code: format_stock_option(code, labels))
    controls[1].metric("股票池", len(codes))

    holding = holding_map_from_table(holdings).get(selected_code)
    with st.spinner(f"正在拉取真实数据：{format_stock_option(selected_code, labels)}"):
        try:
            result = fetch_result(selected_code, labels, days, realtime, refresh_key, "full", holding)
        except Exception as exc:
            st.error(f"{selected_code} 真实数据获取失败：{type(exc).__name__}。系统未使用演示数据，请稍后重试或切换标的。")
            return
    strategies = evaluate_strategy_skills(result)
    dashboard = build_decision_dashboard(result, strategies)

    render_price_chart(result)
    render_decision_card(dashboard)

    render_sniper_points(dashboard)

    report_tab, data_tab, strategy_tab, agent_tab = st.tabs(["详细报告", "数据明细", "策略明细", "多角色复核"])
    with report_tab:
        render_report_blocks(dashboard)
    with data_tab:
        render_component_table(result)
    with strategy_tab:
        render_strategy_table(strategies)
    with agent_tab:
        review = build_agent_review(result)
        render_agent_review(review)


def render_price_chart(result: dict) -> None:
    frame = result.get("frame")
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        st.info("暂无可绘制行情。")
        return
    df = frame.dropna(subset=["date", "open", "high", "low", "close"]).copy()
    if df.empty:
        st.info("行情字段不完整。")
        return
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    chart_df = df.tail(180).copy() if len(df) > 180 else df

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.78, 0.22],
        vertical_spacing=0.02,
    )
    fig.add_trace(
        go.Candlestick(
            x=chart_df["date"],
            open=chart_df["open"],
            high=chart_df["high"],
            low=chart_df["low"],
            close=chart_df["close"],
            name="K线",
            increasing_line_color="#dc2626",
            decreasing_line_color="#16a34a",
            increasing_fillcolor="#dc2626",
            decreasing_fillcolor="#16a34a",
        ),
        row=1,
        col=1,
    )
    for col, color in [("ma5", "#f59e0b"), ("ma20", "#38bdf8"), ("ma60", "#a78bfa")]:
        if col in chart_df:
            fig.add_trace(go.Scatter(x=chart_df["date"], y=chart_df[col], mode="lines", name=col.upper(), line=dict(color=color, width=2.0)), row=1, col=1)
    volume_colors = ["rgba(220,38,38,.62)" if c >= o else "rgba(22,163,74,.62)" for o, c in zip(chart_df["open"], chart_df["close"])]
    fig.add_trace(go.Bar(x=chart_df["date"], y=chart_df.get("volume", pd.Series([0] * len(chart_df))), marker_color=volume_colors, name="成交量"), row=2, col=1)
    levels = result.get("levels") or {}
    for key, label, color in [
        ("conservative_entry", "稳健买点", "#38bdf8"),
        ("breakout_entry", "突破买点", "#ef4444"),
        ("stop_loss", "止损", "#16a34a"),
        ("take_profit_watch", "止盈观察", "#f59e0b"),
    ]:
        value = levels.get(key)
        if value is not None and pd.notna(value):
            fig.add_hline(y=float(value), line_width=1, line_dash="dot", line_color=color, annotation_text=label, row=1, col=1)
    fig.update_layout(
        height=780,
        margin=dict(l=16, r=72, t=54, b=18),
        template="plotly_white",
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        title=f"{result.get('code')} {result.get('name')} / {result.get('action_label')} / {result.get('score'):.1f}",
        legend=dict(orientation="h", yanchor="bottom", y=1.01, x=0, font=dict(size=13)),
        xaxis_rangeslider_visible=False,
        hovermode="x unified",
        font=dict(color="#0f172a", size=13),
        dragmode="pan",
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(148,163,184,.25)", tickfont=dict(size=12))
    fig.update_yaxes(showgrid=True, gridcolor="rgba(148,163,184,.25)", tickfont=dict(size=12), side="right")
    st.plotly_chart(fig, use_container_width=True)


def render_decision_card(dashboard: dict) -> None:
    meta = dashboard["meta"]
    core = dashboard["core_conclusion"]
    data = dashboard["data_perspective"]
    score = float(core.get("score") or 0)
    html = f"""
    <div class="decision-hero">
        <div class="decision-top">
            <div>
                <div class="label-uppercase">DECISION DASHBOARD</div>
                <div class="stock-name">{escape(str(meta.get("name") or meta.get("code")))}</div>
                <div class="stock-meta">{escape(str(meta.get("code")))} / {escape(str(meta.get("industry") or "未知行业"))} / {escape(str(meta.get("latest_trade_date") or "未知交易日"))}</div>
            </div>
            <div class="score-ring" style="--score:{score};">
                <div class="score-ring-inner"><b>{score:.0f}</b><span>SCORE</span></div>
            </div>
        </div>
        <div class="one-sentence">{escape(core.get("one_sentence", ""))}</div>
        <div class="strategy-list" style="grid-template-columns:1fr 1fr;">
            <div class="strategy-item active"><div class="strategy-head"><b>信号</b><span class="strategy-score">{escape(str(core.get("signal_type")))}</span></div><div class="strategy-reason">空仓：{escape(core["position_advice"]["no_position"])}</div></div>
            <div class="strategy-item"><div class="strategy-head"><b>数据质量</b><span class="strategy-score">{data["data_quality"].get("score", "N/A")}</span></div><div class="strategy-reason">{escape(_join(data["data_quality"].get("evidence")))}</div></div>
        </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def render_sniper_points(dashboard: dict) -> None:
    points = dashboard["battle_plan"]["sniper_points"]
    labels = [("ideal_buy", "稳健买点"), ("breakout_buy", "突破买点"), ("stop_loss", "止损位"), ("take_profit", "止盈观察")]
    html = '<div class="sniper-grid">'
    for key, label in labels:
        html += f'<div class="sniper-box"><span>{escape(label)}</span><b>{escape(str(points.get(key, "N/A")))}</b></div>'
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)


def render_strategy_panel(strategies: list[dict]) -> None:
    cards = []
    for item in strategies[:6]:
        cls = "strategy-item"
        if item.get("tone") == "active":
            cls += " active"
        elif item.get("tone") == "risk":
            cls += " risk"
        cards.append(
            f'<div class="{cls}">'
            f'<div class="strategy-head"><b>{escape(str(item["display_name"]))}</b>'
            f'<span class="strategy-score">{escape(str(item["score"]))}</span></div>'
            f'<div class="strategy-reason">{escape(_join(item.get("reasons")))}</div>'
            f"</div>"
        )
    html = (
        '<div class="dsa-card" style="margin-top:12px;">'
        '<div class="label-uppercase">STRATEGY SKILLS</div>'
        f'<div class="strategy-list">{"".join(cards)}</div>'
        "</div>"
    )
    st.markdown(html, unsafe_allow_html=True)


def render_report_blocks(dashboard: dict) -> None:
    core = dashboard["core_conclusion"]
    intel = dashboard["intelligence"]
    plan = dashboard["battle_plan"]
    left, right = st.columns(2)
    with left:
        st.markdown("#### 结论与仓位")
        advice = core["position_advice"]
        st.write(f"- 空仓：{advice['no_position']}")
        st.write(f"- 已持有：{advice['holding']}")
        st.write(f"- 一句话：{core.get('one_sentence', '')}")
        st.markdown("#### 催化因素")
        for item in intel["positive_catalysts"]:
            st.write(f"- {item}")
    with right:
        st.markdown("#### 风险清单")
        for item in intel["risk_alerts"]:
            st.write(f"- {item}")
        st.markdown("#### 操作检查清单")
        for item in plan["action_checklist"]:
            st.write(f"- {item}")


def render_component_table(result: dict) -> None:
    rows = []
    for name, item in (result.get("score_components") or {}).items():
        rows.append(
            {
                "维度": name,
                "评分": item.get("score"),
                "权重": f"{float(item.get('weight', 0)) * 100:.0f}%" if item.get("available", True) else "未计入",
                "依据": _join(item.get("evidence")),
                "风险": _join(item.get("concerns")),
            }
        )
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    metrics = result.get("metrics") or {}
    st.dataframe(pd.DataFrame(metrics.items(), columns=["指标", "数值"]), hide_index=True, use_container_width=True)


def render_strategy_table(strategies: list[dict]) -> None:
    rows = []
    for item in strategies:
        rows.append(
            {
                "策略": item.get("display_name"),
                "分数": item.get("score"),
                "状态": item.get("tone"),
                "触发": "是" if item.get("matched") else "否",
                "依据": _join(item.get("reasons")),
            }
        )
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)


def render_agent_review(review: dict) -> None:
    cols = st.columns(4)
    cols[0].metric("评级", review["rating"])
    cols[1].metric("动作", review["trader_action"])
    cols[2].metric("一致性", review["consensus"])
    cols[3].metric("偏多/偏空", f"{review['bullish_count']}/{review['bearish_count']}")
    rows = []
    for item in review["analysts"]:
        rows.append(
            {
                "角色": item["role"],
                "立场": item["stance"],
                "评分": item["score"],
                "支持": _join(item["evidence"]),
                "担忧": _join(item["concerns"]),
            }
        )
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)


def render_portfolio_page(codes: list[str], labels: dict[str, str], days: int, realtime: bool, refresh_key: int, holdings: pd.DataFrame) -> None:
    if st.button("扫描组合", type="primary"):
        st.session_state["portfolio_scan_requested"] = True
    if not st.session_state.get("portfolio_scan_requested", True):
        st.info("点击扫描组合开始。")
        return
    holding_map = holding_map_from_table(holdings)
    progress = st.progress(0, text="准备扫描组合...")
    results = []
    for idx, code in enumerate(codes):
        progress.progress(idx / max(len(codes), 1), text=f"拉取 {idx + 1}/{len(codes)}：{format_stock_option(code, labels)}")
        try:
            result = fetch_result(code, labels, days, realtime, refresh_key, "scan", holding_map.get(code))
            results.append(result)
        except Exception as exc:
            st.error(f"{code} 真实数据获取失败：{type(exc).__name__}，已跳过。")
    progress.empty()
    results = sorted(results, key=lambda item: (item["rank"], -item["score"]))
    brief = build_market_brief(results)
    cols = st.columns(4)
    cols[0].metric("扫描数量", len(results))
    cols[1].metric("买入/试探", brief["buy_count"])
    cols[2].metric("持有/观察", brief["watch_count"])
    cols[3].metric("减仓/回避", brief["risk_count"], f"均分 {brief['avg_score']:.1f}")
    st.dataframe(make_portfolio_table(results), hide_index=True, use_container_width=True, height=min(520, 80 + 34 * max(len(results), 2)))


def make_portfolio_table(results: list[dict]) -> pd.DataFrame:
    rows = []
    for result in results:
        components = result.get("score_components") or {}
        levels = result.get("levels") or {}
        rows.append(
            {
                "代码": result["code"],
                "名称": result["name"],
                "操作": result["action_label"],
                "分数": round(result["score"], 1),
                "趋势": components.get("趋势结构", {}).get("score"),
                "动能": components.get("动能强度", {}).get("score"),
                "数据质量": components.get("数据质量", {}).get("score"),
                "现价": round(result["last_close"], 2),
                "稳健买点": round(levels.get("conservative_entry", 0), 2),
                "止损": round(levels.get("stop_loss", 0), 2),
                "核心结论": result.get("summary", ""),
            }
        )
    return pd.DataFrame(rows)


def render_discovery_page(days: int, realtime: bool, refresh_key: int) -> None:
    if st.session_state.get("discovery_schema_version") != DISCOVERY_SCHEMA_VERSION:
        for key in ("discovery_candidates", "discovery_source", "discovery_deep_count"):
            st.session_state.pop(key, None)
        st.session_state["discovery_schema_version"] = DISCOVERY_SCHEMA_VERSION

    modes = list(MODE_LABELS.keys())
    with st.form("discovery_form"):
        cols = st.columns(4)
        mode = cols[0].selectbox("筛选风格", modes, format_func=lambda key: MODE_LABELS[key])
        limit = cols[1].slider("候选数", min_value=20, max_value=120, value=60, step=10)
        show_count = cols[2].slider("展示", min_value=5, max_value=30, value=15, step=5)
        deep_count = cols[3].slider("深度复核", min_value=0, max_value=10, value=0, step=1)
        submitted = st.form_submit_button("实时拉取市场快照并筛选", type="primary", use_container_width=True)
    st.caption(MODE_DESCRIPTIONS[mode])
    if not submitted and "discovery_candidates" not in st.session_state:
        st.info("等待筛选。")
        return
    if submitted:
        with st.spinner("正在实时拉取全市场快照，本次不读取本地缓存..."):
            snapshot, source, warnings = load_market_snapshot_fresh(force_refresh=True)
        if "不可用" in source or snapshot.empty:
            reason = f"：{warnings[0]}" if warnings else ""
            st.error(f"真实全市场快照不可用{reason}。已停止筛选，不使用演示候选池。")
            return
        candidates = screen_market_candidates(snapshot, mode=mode, limit=limit)
        if isinstance(candidates, pd.DataFrame) and not candidates.empty and "行情初筛" in set(candidates.get("复核状态", pd.Series(dtype=str)).astype(str)):
            with st.spinner("正在用真实单股历史行情补全可计算字段..."):
                candidates, enrich_warnings = enrich_candidates_with_history(
                    candidates,
                    mode=mode,
                    days=days,
                    realtime=False,
                    limit=show_count,
                )
            if enrich_warnings:
                st.caption(f"部分候选无法补全历史字段：{len(enrich_warnings)} 只。")
        st.session_state["discovery_candidates"] = candidates.head(show_count)
        st.session_state["discovery_source"] = source
        st.session_state["discovery_deep_count"] = deep_count
        st.session_state["discovery_schema_version"] = DISCOVERY_SCHEMA_VERSION
    source = st.session_state.get("discovery_source", "")
    candidates = st.session_state.get("discovery_candidates", pd.DataFrame())
    st.caption(f"数据源：{source}")
    if isinstance(candidates, pd.DataFrame) and candidates.empty:
        st.warning("当前真实快照没有命中候选。系统不会使用演示数据或补造候选。")
    elif isinstance(candidates, pd.DataFrame) and "行情初筛" in set(candidates.get("复核状态", pd.Series(dtype=str)).astype(str)):
        st.info("当前全市场快照不含估值/市值字段，已用真实单股历史行情补全可计算字段；PE/PB、市值不会补造。")
    st.dataframe(make_discovery_table(candidates), hide_index=True, use_container_width=True, height=430)

    deep_count = int(st.session_state.get("discovery_deep_count", 0))
    if deep_count > 0 and isinstance(candidates, pd.DataFrame) and not candidates.empty:
        labels = {}
        rows = []
        for code in candidates["代码"].head(deep_count).astype(str).tolist():
            try:
                result = fetch_result(code, labels, days, realtime, refresh_key, "analysis")
                strategies = evaluate_strategy_skills(result)
                rows.append(
                    {
                        "代码": result["code"],
                        "名称": result["name"],
                        "操作": result["action_label"],
                        "综合分": round(result["score"], 1),
                        "触发策略": "、".join([item["display_name"] for item in strategies if item["matched"] and item["tone"] != "risk"][:3]),
                        "核心": result["summary"],
                    }
                )
            except Exception as exc:
                rows.append({"代码": code, "名称": "", "操作": "数据失败", "综合分": 0, "触发策略": "", "核心": type(exc).__name__})
        st.markdown("#### 深度复核")
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)


def make_discovery_table(candidates: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(candidates, pd.DataFrame) or candidates.empty:
        return pd.DataFrame()
    base_cols = ["代码", "名称", "最新价", "涨跌幅", "成交额", "初筛分", "复核状态", "可用增强字段", "初筛理由"]
    optional_cols = [
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
    ordered_cols = []
    for col in base_cols:
        if col in candidates.columns:
            ordered_cols.append(col)
    for col in optional_cols:
        if col in candidates.columns and candidates[col].notna().any():
            ordered_cols.insert(max(len(ordered_cols) - 3, 0), col)
    table = candidates.loc[:, ordered_cols].copy()
    for col in ["成交额", "总市值", "流通市值", "主力净流入"]:
        if col in table:
            table[col] = table[col].apply(format_table_money)
    for col in ["最新价", "涨跌幅", "换手率", "市盈率-动态", "市净率", "量比", "5分钟涨跌", "60日涨跌幅", "年初至今涨跌幅", "市盈率TTM", "振幅", "初筛分"]:
        if col in table:
            table[col] = pd.to_numeric(table[col], errors="coerce").round(2)
    return table


def format_table_money(value) -> str:
    if value is None or pd.isna(value):
        return ""
    value = float(value)
    if abs(value) >= 1e8:
        return f"{value / 1e8:.2f}亿"
    if abs(value) >= 1e4:
        return f"{value / 1e4:.1f}万"
    return f"{value:.0f}"


def render_backtest_page(codes: list[str], labels: dict[str, str], days: int) -> None:
    selected_code = st.selectbox("回测标的", codes, format_func=lambda code: format_stock_option(code, labels))
    with st.spinner("正在拉取历史行情..."):
        try:
            result = fetch_result(selected_code, labels, days, False, 0, "history")
        except Exception as exc:
            st.error(f"{selected_code} 历史行情获取失败：{type(exc).__name__}。未使用演示数据。")
            return
    bt = run_ma_backtest(result["frame"])
    summary = bt["summary"]
    if "样本不足" in summary:
        st.warning(summary["样本不足"])
        return
    cols = st.columns(6)
    cols[0].metric("策略收益", f"{summary['策略收益'] * 100:.1f}%")
    cols[1].metric("买入持有", f"{summary['买入持有收益'] * 100:.1f}%")
    cols[2].metric("最大回撤", f"{summary['最大回撤'] * 100:.1f}%")
    cols[3].metric("交易次数", summary["交易次数"])
    cols[4].metric("胜率", f"{summary['胜率'] * 100:.1f}%")
    cols[5].metric("简化夏普", f"{summary['简化夏普']:.2f}")
    equity = bt["equity"]
    fig = go.Figure()
    if not equity.empty:
        fig.add_trace(go.Scatter(x=equity["date"], y=equity["equity"], mode="lines", name="策略权益", line=dict(color="#0891b2", width=2)))
    fig.update_layout(height=380, template="plotly_white", margin=dict(l=8, r=8, t=28, b=8))
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(bt["trades"], hide_index=True, use_container_width=True)


def render_rules_page() -> None:
    st.markdown("#### 策略能力包")
    st.dataframe(pd.DataFrame([skill.__dict__ for skill in STRATEGY_LIBRARY]), hide_index=True, use_container_width=True)
    st.markdown("#### 数据真实性约束")
    st.write("系统只使用真实行情、资金、财务、新闻和公告接口；接口不可用时显示警告或停止筛选，不使用演示数据。")
    st.markdown("#### 参考项目")
    st.write("本版工作台重点参考 `daily_stock_analysis-main` 的工作流：报告优先、策略能力包、历史/任务/组合/回测分区，以及 MIT 许可项目的架构思想。")


def _join(values) -> str:
    if not values:
        return "暂无"
    return "；".join(str(value) for value in values if str(value).strip()) or "暂无"
