from __future__ import annotations

import os
import time
from datetime import date, datetime
from html import escape
from pathlib import Path
from urllib.parse import quote

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

try:
    from streamlit_autorefresh import st_autorefresh
except Exception:
    st_autorefresh = None

from src.ashare_advisor.agents import build_agent_markdown, build_agent_review
from src.ashare_advisor.backtest import run_ma_backtest
from src.ashare_advisor.data import DataProvider, normalize_codes
from src.ashare_advisor.reporting import build_rule_report
from src.ashare_advisor.rules import analyze_stock, build_market_brief
from src.ashare_advisor.screener import (
    MODE_DESCRIPTIONS,
    MODE_LABELS,
    clear_market_snapshot_file_cache,
    load_market_snapshot,
    screen_market_candidates,
)


st.set_page_config(
    page_title="A股个人投资操作助手",
    page_icon="📈",
    layout="wide",
)


def get_config_value(name: str, default: str = "") -> str:
    env_value = os.getenv(name)
    if env_value:
        return env_value
    local_secret_paths = [
        Path.cwd() / ".streamlit" / "secrets.toml",
        Path.home() / ".streamlit" / "secrets.toml",
    ]
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
        try:
            values = st.experimental_get_query_params().get(name, [default])
            value = values[0] if values else default
        except Exception:
            value = default
    if isinstance(value, list):
        return str(value[0]) if value else default
    return str(value) if value is not None else default


def set_query_value(name: str, value: str) -> None:
    try:
        st.query_params[name] = value
    except Exception:
        st.experimental_set_query_params(**{name: value})


def require_login() -> None:
    expected_password = get_config_value("APP_PASSWORD")
    if not expected_password:
        return
    if st.session_state.get("authenticated"):
        return

    st.title("A股个人投资操作助手")
    st.caption("请输入访问密码。仅供个人研究，不构成投资建议。")
    password = st.text_input("访问密码", type="password")
    if password:
        if password == expected_password:
            st.session_state["authenticated"] = True
            st.rerun()
        else:
            st.error("密码不正确")
    st.stop()


require_login()


def inject_theme() -> None:
    st.markdown(
        """
        <style>
        :root {
            --surface: #ffffff;
            --page: #f3f5f8;
            --line: #dde3ea;
            --line-strong: #cbd5e1;
            --ink: #0f172a;
            --muted: #64748b;
            --primary: #ef4444;
            --primary-soft: #fff1f2;
            --green: #16a34a;
            --terminal: #111827;
            --terminal-2: #1f2937;
        }

        .stApp {
            background: var(--page);
            color: var(--ink);
        }

        .block-container {
            max-width: 1720px;
            padding-top: 1.2rem;
            padding-bottom: 3rem;
        }

        section[data-testid="stSidebar"] {
            background: #edf1f6;
            border-right: 1px solid var(--line);
        }

        section[data-testid="stSidebar"] h1,
        section[data-testid="stSidebar"] h2,
        section[data-testid="stSidebar"] h3 {
            letter-spacing: 0;
        }

        section[data-testid="stSidebar"] textarea,
        section[data-testid="stSidebar"] input {
            border-radius: 7px;
        }

        section[data-testid="stSidebar"] [data-testid="stDataFrame"] {
            border-color: #cfd8e3;
        }

        section[data-testid="stSidebar"] textarea {
            background: #ffffff;
            border: 1px solid #d7dee8;
            box-shadow: inset 0 1px 2px rgba(15, 23, 42, 0.04);
            min-height: 112px !important;
        }

        section[data-testid="stSidebar"] div[data-testid="stButton"] button {
            background: #ffffff;
            border: 1px solid #cbd5e1;
        }

        section[data-testid="stSidebar"] div[data-testid="stButton"] button:hover {
            border-color: #64748b;
            color: #0f172a;
        }

        .side-console {
            background: #ffffff;
            border: 1px solid #dbe3ef;
            border-radius: 10px;
            box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
            margin: 0 0 14px;
            padding: 12px 13px;
        }

        .side-console-title {
            color: #0f172a;
            font-size: 18px;
            font-weight: 850;
            margin-bottom: 4px;
        }

        .side-console-sub {
            color: #64748b;
            font-size: 12px;
            line-height: 1.5;
        }

        .side-panel {
            background: rgba(255, 255, 255, 0.86);
            border: 1px solid #dbe3ef;
            border-radius: 12px;
            box-shadow: 0 10px 24px rgba(15, 23, 42, 0.05);
            margin: 0 0 10px;
            padding: 10px 11px;
        }

        .side-panel-head {
            align-items: center;
            display: flex;
            justify-content: space-between;
            gap: 10px;
            margin-bottom: 8px;
        }

        .side-panel-title {
            color: #0f172a;
            font-size: 14px;
            font-weight: 850;
            letter-spacing: 0;
        }

        .side-panel-kicker {
            color: #ef4444;
            font-size: 10px;
            font-weight: 850;
            letter-spacing: .08em;
            text-transform: uppercase;
        }

        .side-panel-copy {
            color: #64748b;
            font-size: 12px;
            line-height: 1.55;
            margin: 0 0 10px;
        }

        .side-compact-label {
            color: #475569;
            font-size: 12px;
            font-weight: 800;
            margin: 6px 0 6px;
        }

        .side-token-row {
            display: flex;
            flex-wrap: wrap;
            gap: 6px;
            margin: 8px 0 10px;
            min-height: 28px;
        }

        .side-token {
            background: #f8fafc;
            border: 1px solid #dbe3ef;
            border-radius: 999px;
            color: #0f172a;
            font-size: 12px;
            font-weight: 750;
            padding: 5px 8px;
            transition: border-color .15s ease, box-shadow .15s ease, transform .15s ease;
        }

        .side-token:hover {
            border-color: #ef4444;
            box-shadow: 0 8px 18px rgba(239, 68, 68, 0.12);
            transform: translateY(-1px);
        }

        .side-token-muted {
            color: #64748b;
            font-weight: 650;
        }

        .side-metric-grid {
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 8px;
            margin: 8px 0 10px;
        }

        .side-metric {
            background: #f8fafc;
            border: 1px solid #e2e8f0;
            border-radius: 8px;
            padding: 8px;
        }

        .side-metric-label {
            color: #64748b;
            font-size: 11px;
            margin-bottom: 3px;
        }

        .side-metric-value {
            color: #0f172a;
            font-size: 16px;
            font-weight: 850;
        }

        .side-hint {
            background: #f8fafc;
            border: 1px solid #e2e8f0;
            border-left: 3px solid #2563eb;
            border-radius: 8px;
            color: #475569;
            font-size: 12px;
            line-height: 1.55;
            margin: 8px 0 10px;
            padding: 8px 9px;
        }

        .side-rule {
            border-top: 1px solid #dbe3ef;
            margin: 12px 0 10px;
        }

        section[data-testid="stSidebar"] div[data-testid="stExpander"] {
            background: rgba(255, 255, 255, 0.78);
            border: 1px solid #dbe3ef;
            border-radius: 12px;
            box-shadow: 0 8px 20px rgba(15, 23, 42, 0.04);
        }

        section[data-testid="stSidebar"] div[data-testid="stExpander"] summary {
            color: #0f172a;
            font-weight: 850;
        }

        .app-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 24px;
            background: linear-gradient(135deg, var(--terminal), var(--terminal-2));
            border: 1px solid #0b1220;
            border-radius: 8px;
            box-shadow: 0 10px 24px rgba(15, 23, 42, 0.14);
            margin-bottom: 12px;
            padding: 18px 20px;
        }

        .app-eyebrow {
            color: #fca5a5;
            font-size: 12px;
            font-weight: 700;
            letter-spacing: 0.08em;
            margin-bottom: 7px;
        }

        .app-header h1 {
            margin: 0;
            color: #f8fafc;
            font-size: 28px;
            line-height: 1.2;
            letter-spacing: 0;
        }

        .app-header p {
            margin: 8px 0 0;
            color: #cbd5e1;
            font-size: 14px;
        }

        .header-pills {
            display: flex;
            flex-wrap: wrap;
            justify-content: flex-end;
            gap: 8px;
            min-width: 360px;
        }

        .header-pill {
            background: rgba(255, 255, 255, 0.08);
            border: 1px solid rgba(255, 255, 255, 0.14);
            border-radius: 6px;
            color: #e2e8f0;
            font-size: 13px;
            font-weight: 600;
            padding: 8px 10px;
            white-space: nowrap;
        }

        .section-note {
            background: #ffffff;
            border: 1px solid var(--line);
            border-left: 4px solid #2563eb;
            border-radius: 8px;
            color: #334155;
            font-size: 14px;
            line-height: 1.7;
            margin: 8px 0 16px;
            padding: 10px 13px;
        }

        .page-header {
            align-items: flex-end;
            border-bottom: 1px solid var(--line);
            display: flex;
            justify-content: space-between;
            gap: 18px;
            margin: 18px 0 14px;
            padding-bottom: 10px;
        }

        .page-header h2 {
            color: var(--ink);
            font-size: 24px;
            line-height: 1.25;
            letter-spacing: 0;
            margin: 0;
        }

        .page-header p {
            color: var(--muted);
            font-size: 13px;
            margin: 6px 0 0;
        }

        .page-tag {
            background: #f8fafc;
            border: 1px solid var(--line);
            border-radius: 6px;
            color: #475569;
            font-size: 12px;
            font-weight: 700;
            padding: 6px 9px;
            white-space: nowrap;
        }

        div[data-testid="stMetric"] {
            background: var(--surface);
            border: 1px solid var(--line);
            border-radius: 8px;
            box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
            padding: 12px 14px;
        }

        div[data-testid="stMetricLabel"] p {
            color: var(--muted);
            font-size: 13px;
        }

        div[data-testid="stMetricValue"] {
            color: var(--ink);
            font-size: 26px;
            line-height: 1.1;
        }

        div[data-testid="stDataFrame"] {
            background: var(--surface);
            border: 1px solid var(--line);
            border-radius: 8px;
            overflow: hidden;
        }

        div[data-testid="stRadio"] [role="radiogroup"] {
            display: flex;
            flex-wrap: wrap;
            gap: 0;
            margin: 0 0 18px;
            background: #ffffff;
            border: 1px solid var(--line);
            border-radius: 8px;
            padding: 4px;
        }

        div[data-testid="stRadio"] [role="radiogroup"] label {
            background: transparent;
            border: 1px solid transparent;
            border-radius: 6px;
            box-shadow: none;
            color: #334155;
            min-height: 32px;
            padding: 5px 11px;
        }

        div[data-testid="stRadio"] [role="radiogroup"] label:has(input:checked) {
            background: var(--primary-soft);
            border-color: var(--primary);
            color: #b91c1c;
            font-weight: 700;
        }

        div[data-testid="stRadio"] [role="radiogroup"] label > div:first-child {
            display: none;
        }

        div[data-testid="stButton"] button,
        div[data-testid="stDownloadButton"] button {
            border-radius: 7px;
            font-weight: 650;
            min-height: 36px;
        }

        .action-card {
            background: var(--surface);
            border: 1px solid var(--line);
            border-left-width: 5px;
            border-radius: 8px;
            box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
            min-height: 108px;
            padding: 13px 15px;
        }

        .action-card-meta {
            color: var(--muted);
            font-size: 13px;
            line-height: 1.4;
            margin-bottom: 6px;
        }

        .action-card-title {
            font-size: 24px;
            font-weight: 800;
            line-height: 1.2;
            margin-bottom: 8px;
        }

        .action-card-score {
            color: #334155;
            font-size: 14px;
            line-height: 1.5;
        }

        .stAlert {
            border-radius: 8px;
        }

        h3 {
            color: var(--ink);
            font-size: 18px;
            letter-spacing: 0;
            margin-top: 1.15rem;
        }

        h4 {
            color: #1e293b;
            font-size: 15px;
            letter-spacing: 0;
        }

        header[data-testid="stHeader"] {
            background: transparent;
            height: 0;
        }

        div[data-testid="stToolbar"],
        div[data-testid="stDecoration"],
        div[data-testid="stStatusWidget"],
        .stDeployButton,
        #MainMenu,
        footer {
            display: none !important;
            height: 0 !important;
        }

        .main .block-container {
            max-width: none !important;
            padding: 1.25rem 2rem 3rem !important;
        }

        section[data-testid="stSidebar"] {
            min-width: 340px !important;
            width: 340px !important;
        }

        section[data-testid="stSidebar"] > div {
            padding-top: 1.2rem;
            width: 340px !important;
        }

        .app-header {
            background: #ffffff;
            border: 1px solid var(--line);
            border-top: 3px solid #111827;
            box-shadow: 0 1px 2px rgba(15, 23, 42, 0.05);
            margin: 0 0 12px;
            min-height: 92px;
            overflow: visible;
            padding: 16px 18px;
        }

        .app-eyebrow {
            color: #b91c1c;
            font-size: 11px;
            line-height: 1.2;
            margin-bottom: 8px;
        }

        .app-header h1 {
            color: var(--ink);
            font-size: 26px;
        }

        .app-header p {
            color: var(--muted);
            max-width: 760px;
        }

        .header-pills {
            min-width: 0;
            max-width: 620px;
        }

        .header-pill {
            background: #f8fafc;
            border-color: var(--line);
            color: #334155;
        }

        .page-header {
            margin-top: 14px;
        }

        .action-card-reasons {
            border-top: 1px solid #eef2f7;
            color: #334155;
            font-size: 13px;
            line-height: 1.6;
            margin-top: 10px;
            padding-top: 9px;
        }

        .action-card-reasons div {
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
        }

        .scope-panel {
            display: grid;
            grid-template-columns: minmax(0, 2fr) minmax(280px, 1fr);
            gap: 12px;
            margin: 0 0 16px;
        }

        .scope-card {
            background: #ffffff;
            border: 1px solid var(--line);
            border-radius: 8px;
            padding: 12px 14px;
            transition: border-color .15s ease, box-shadow .15s ease;
        }

        .scope-card:hover {
            border-color: #94a3b8;
            box-shadow: 0 10px 22px rgba(15, 23, 42, 0.08);
        }

        .scope-title {
            color: #475569;
            font-size: 12px;
            font-weight: 800;
            letter-spacing: 0.06em;
            margin-bottom: 9px;
        }

        .scope-chips {
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
        }

        .scope-chip {
            background: #f8fafc;
            border: 1px solid #e2e8f0;
            border-radius: 6px;
            color: #0f172a;
            font-size: 13px;
            font-weight: 650;
            padding: 6px 8px;
            transition: background .15s ease, border-color .15s ease, transform .15s ease;
        }

        .scope-chip:hover {
            background: #fff1f2;
            border-color: #ef4444;
            transform: translateY(-1px);
        }

        .scope-stat-row {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 8px;
        }

        .scope-stat {
            background: #f8fafc;
            border: 1px solid #e2e8f0;
            border-radius: 6px;
            padding: 8px;
        }

        .scope-stat-label {
            color: #64748b;
            font-size: 12px;
            margin-bottom: 4px;
        }

        .scope-stat-value {
            color: #0f172a;
            font-size: 17px;
            font-weight: 800;
        }

        .scope-empty {
            color: #64748b;
            font-size: 13px;
            line-height: 1.6;
        }

        .smart-nav {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 12px;
            margin: 0 0 16px;
            position: relative;
            z-index: 20;
        }

        .smart-group {
            background: rgba(255, 255, 255, 0.92);
            border: 1px solid var(--line);
            border-radius: 10px;
            box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
            min-height: 76px;
            padding: 13px 14px;
            position: relative;
            transition: border-color .15s ease, box-shadow .15s ease, transform .15s ease;
        }

        .smart-group:hover {
            border-color: #94a3b8;
            box-shadow: 0 14px 30px rgba(15, 23, 42, 0.12);
            transform: translateY(-1px);
        }

        .smart-group-active {
            border-color: #ef4444;
            box-shadow: 0 10px 24px rgba(239, 68, 68, 0.12);
        }

        .smart-group-label {
            color: #0f172a;
            font-size: 16px;
            font-weight: 850;
            margin-bottom: 5px;
        }

        .smart-group-desc {
            color: #64748b;
            font-size: 12px;
            line-height: 1.45;
        }

        .smart-flyout {
            background: #ffffff;
            border: 1px solid #cbd5e1;
            border-radius: 10px;
            box-shadow: 0 18px 42px rgba(15, 23, 42, 0.18);
            left: 10px;
            opacity: 0;
            padding: 8px;
            pointer-events: none;
            position: absolute;
            right: 10px;
            top: calc(100% + 8px);
            transform: translateY(-4px);
            transition: opacity .15s ease, transform .15s ease;
            z-index: 100;
        }

        .smart-group:hover .smart-flyout {
            opacity: 1;
            pointer-events: auto;
            transform: translateY(0);
        }

        .smart-link {
            border-radius: 8px;
            color: #0f172a !important;
            display: block;
            padding: 9px 10px;
            text-decoration: none !important;
        }

        .smart-link:hover {
            background: #f1f5f9;
        }

        .smart-link-active {
            background: #fff1f2;
            color: #b91c1c !important;
            font-weight: 800;
        }

        .smart-link-title {
            display: block;
            font-size: 13px;
            font-weight: 800;
        }

        .smart-link-desc {
            color: #64748b;
            display: block;
            font-size: 12px;
            line-height: 1.45;
            margin-top: 2px;
        }

        @media (max-width: 900px) {
            .main .block-container {
                padding: 1rem 1rem 2rem !important;
            }

            section[data-testid="stSidebar"],
            section[data-testid="stSidebar"] > div {
                width: 100% !important;
                min-width: 0 !important;
            }

            .app-header {
                align-items: flex-start;
                flex-direction: column;
                min-height: 0;
            }

            .header-pills {
                justify-content: flex-start;
                min-width: 0;
            }

            .app-header h1 {
                font-size: 28px;
            }

            .scope-panel {
                grid-template-columns: 1fr;
            }

            .smart-nav {
                grid-template-columns: 1fr;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_app_header(code_count: int, days: int, realtime: bool, auto_refresh: bool, refresh_seconds: int) -> None:
    realtime_text = "分钟线开启" if realtime else "快速日线模式"
    refresh_text = f"{refresh_seconds} 秒自动刷新" if realtime and auto_refresh else "手动刷新"
    st.markdown(
        f"""
        <div class="app-header">
            <div>
                <div class="app-eyebrow">A-SHARE RESEARCH DESK</div>
                <h1>A股个人投资操作助手</h1>
                <p>自选池监控、结构分析、风险线和候选发现集中在一个操作台。仅供研究，不构成投资建议。</p>
            </div>
            <div class="header-pills">
                <span class="header-pill">WATCHLIST {code_count}</span>
                <span class="header-pill">WINDOW {days}D</span>
                <span class="header-pill">{escape(realtime_text)}</span>
                <span class="header-pill">{escape(refresh_text)}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_smart_nav(current_page: str, codes: list[str]) -> None:
    encoded_codes = quote(",".join(codes))
    groups = [
        (
            "组合工作台",
            "股票池、行业暴露、每日动作合在一个操作流。",
            [
                ("自选股扫描", "快速看买卖信号、风险和关键价位"),
                ("板块轮动", "市场板块与自选池暴露对照"),
                ("每日操作清单", "把当天动作整理成待办清单"),
            ],
        ),
        (
            "个股研究",
            "从价格结构到资金、财务和回测逐层展开。",
            [
                ("个股分析", "单股操作计划、风险线和指标快照"),
                ("资金财务新闻", "资金流、财务指标、新闻公告"),
                ("简单回测", "验证均线规则过去是否有效"),
                ("多智能体研判", "多角度交叉复核单股观点"),
            ],
        ),
        (
            "发现与规则",
            "从市场里找候选，再看系统为什么这么判断。",
            [
                ("优质股票发现", "全市场粗筛和深度复核"),
                ("规则说明", "打分、风险和动作规则解释"),
            ],
        ),
    ]
    cards = []
    for title, desc, links in groups:
        group_active = any(page == current_page for page, _ in links)
        link_html = []
        for page, link_desc in links:
            active = " smart-link-active" if page == current_page else ""
            href = f"?view={quote(page)}&codes={encoded_codes}"
            link_html.append(
                f"""
                <a class="smart-link{active}" href="{href}">
                    <span class="smart-link-title">{escape(page)}</span>
                    <span class="smart-link-desc">{escape(link_desc)}</span>
                </a>
                """
            )
        cards.append(
            f"""
            <div class="smart-group{' smart-group-active' if group_active else ''}">
                <div class="smart-group-label">{escape(title)}</div>
                <div class="smart-group-desc">{escape(desc)}</div>
                <div class="smart-flyout">{''.join(link_html)}</div>
            </div>
            """
        )
    st.markdown(f'<div class="smart-nav">{"".join(cards)}</div>', unsafe_allow_html=True)


def render_workflow_nav(current_page: str, codes: list[str]) -> None:
    groups = [
        (
            "组合监控",
            "适合合并：股票池扫描、行业暴露和每日操作本来就是同一个组合管理流程。",
            [
                ("自选股扫描", "先看股票池整体买卖信号和风险分布"),
                ("板块轮动", "再看市场行业和自选池行业暴露是否匹配"),
                ("每日操作清单", "最后整理今天要关注、减仓或继续观察的动作"),
            ],
        ),
        (
            "单股研究",
            "适合合并：单只股票的技术、资金、财务、新闻和回测需要上下文连着看。",
            [
                ("个股分析", "操作计划、关键价位、风险提示"),
                ("资金财务新闻", "资金流、财务质量、新闻公告"),
                ("简单回测", "验证当前规则过去是否有效"),
                ("多智能体研判", "多角度复核单股观点"),
            ],
        ),
        (
            "候选发现",
            "保持独立：全市场发现和规则说明是研究入口，不应塞进日常看盘流程。",
            [
                ("优质股票发现", "从全市场缩小研究范围"),
                ("规则说明", "查看系统如何打分和判断风险"),
            ],
        ),
    ]
    cols = st.columns([1, 1, 1, 1.4])
    for idx, (title, desc, items) in enumerate(groups):
        active = any(page == current_page for page, _ in items)
        with cols[idx]:
            label = f"{'● ' if active else ''}{title}"
            with st.popover(label, use_container_width=True):
                st.caption(desc)
                for page, help_text in items:
                    button_type = "primary" if page == current_page else "secondary"
                    if st.button(page, key=f"workflow_nav_{page}", help=help_text, type=button_type, use_container_width=True):
                        set_query_value("view", page)
                        set_query_value("codes", ",".join(codes))
                        st.rerun()
                    st.caption(help_text)
    with cols[3]:
        st.caption("工作流入口只合并有连续关系的功能；个别研究页仍保持独立，避免把所有内容硬塞到一屏。")


def render_page_header(title: str, subtitle: str = "", tag: str = "") -> None:
    subtitle_html = f"<p>{escape(subtitle)}</p>" if subtitle else ""
    tag_html = f'<span class="page-tag">{escape(tag)}</span>' if tag else ""
    st.markdown(
        f"""
        <div class="page-header">
            <div>
                <h2>{escape(title)}</h2>
                {subtitle_html}
            </div>
            {tag_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_section_note(text: str) -> None:
    st.markdown(f'<div class="section-note">{escape(text)}</div>', unsafe_allow_html=True)


def render_research_scope(codes: list[str], labels: dict[str, str], holdings: pd.DataFrame) -> None:
    chips = []
    for code in codes[:12]:
        name = labels.get(code, "")
        label = f"{code} {name}" if name and name != code else code
        chips.append(f'<span class="scope-chip">{escape(label)}</span>')
    if len(codes) > 12:
        chips.append(f'<span class="scope-chip">+{len(codes) - 12}</span>')
    chip_html = "".join(chips) if chips else '<span class="scope-empty">暂无股票池</span>'

    holding_rows = []
    if isinstance(holdings, pd.DataFrame) and not holdings.empty:
        for _, row in holdings.dropna(subset=["code"]).iterrows():
            normalized = normalize_codes([row.get("code", "")])
            if normalized:
                holding_rows.append(
                    {
                        "code": normalized[0],
                        "cost": row.get("cost", 0) or 0,
                        "weight": row.get("weight", 0) or 0,
                    }
                )
    holding_count = len(holding_rows)
    total_weight = sum(float(item["weight"] or 0) for item in holding_rows)
    cost_count = sum(1 for item in holding_rows if float(item["cost"] or 0) > 0)
    holding_html = (
        f"""
        <div class="scope-stat-row">
            <div class="scope-stat">
                <div class="scope-stat-label">持仓股票</div>
                <div class="scope-stat-value">{holding_count}</div>
            </div>
            <div class="scope-stat">
                <div class="scope-stat-label">仓位合计</div>
                <div class="scope-stat-value">{total_weight:.0f}%</div>
            </div>
            <div class="scope-stat">
                <div class="scope-stat-label">成本记录</div>
                <div class="scope-stat-value">{cost_count}</div>
            </div>
        </div>
        """
        if holding_count
        else '<div class="scope-empty">左侧填写持仓成本和仓位后，这里会同步显示组合暴露。</div>'
    )
    st.markdown(
        f"""
        <div class="scope-panel">
            <div class="scope-card">
                <div class="scope-title">CURRENT STOCK POOL</div>
                <div class="scope-chips">{chip_html}</div>
            </div>
            <div class="scope-card">
                <div class="scope-title">POSITION SNAPSHOT</div>
                {holding_html}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


inject_theme()


@st.cache_data(show_spinner=False, ttl=60 * 5)
def load_stock(
    code: str,
    days: int,
    realtime: bool,
    refresh_key: int = 0,
    detail_level: str = "full",
) -> tuple[pd.DataFrame, dict, str]:
    profile_cache_version = "real-data-only-v1"
    provider = DataProvider()
    _ = profile_cache_version
    return provider.load_stock(code, days=days, realtime=realtime, detail_level=detail_level)


@st.cache_data(show_spinner=False, ttl=60 * 60)
def load_boards() -> tuple[pd.DataFrame, str]:
    provider = DataProvider()
    return provider.load_industry_boards()


@st.cache_data(show_spinner=False, ttl=60 * 60)
def load_code_labels(codes: tuple[str, ...]) -> dict[str, str]:
    provider = DataProvider()
    return provider.load_code_name_map(codes)


@st.cache_data(show_spinner=False, ttl=60 * 10)
def load_market_snapshot_cached(refresh_key: int = 0) -> tuple[pd.DataFrame, str, list[str]]:
    return load_market_snapshot()


def fetch_stock_with_progress(
    code: str,
    days: int,
    realtime: bool,
    refresh_key: int,
    detail_level: str,
    label: str,
) -> tuple[pd.DataFrame, dict, str]:
    progress = st.progress(0, text=f"准备拉取 {label} 数据...")
    progress.progress(18, text=f"正在连接数据源：{label}")
    df, profile, source = load_stock(code, days, realtime, refresh_key, detail_level)
    progress.progress(78, text=f"正在计算技术指标：{label}")
    progress.progress(100, text=f"数据已完成：{label}")
    time.sleep(0.15)
    progress.empty()
    return df, profile, source


def fetch_boards_with_progress() -> tuple[pd.DataFrame, str]:
    progress = st.progress(0, text="正在拉取行业板块数据...")
    progress.progress(45, text="正在连接板块行情接口...")
    boards, source = load_boards()
    progress.progress(100, text="板块数据已完成")
    time.sleep(0.15)
    progress.empty()
    return boards, source


def fetch_market_snapshot_with_progress(refresh_key: int = 0) -> tuple[pd.DataFrame, str, list[str]]:
    progress = st.progress(0, text="正在拉取全市场快照...")
    progress.progress(25, text="正在获取全市场行情与基础指标...")
    snapshot, source, warnings = load_market_snapshot_cached(refresh_key)
    progress.progress(75, text=f"已获取 {len(snapshot)} 只股票，正在准备筛选...")
    progress.progress(100, text="全市场快照已完成")
    time.sleep(0.15)
    progress.empty()
    return snapshot, source, warnings


FINANCIAL_FIELDS = [
    ("report_date", "最新财报期", ""),
    ("roe", "净资产收益率ROE", "%"),
    ("weighted_roe", "加权净资产收益率", "%"),
    ("revenue_growth", "营收同比增长", "%"),
    ("profit_growth", "净利润同比增长", "%"),
    ("debt_ratio", "资产负债率", "%"),
    ("gross_margin", "销售毛利率", "%"),
    ("net_margin", "销售净利率", "%"),
    ("cashflow_per_share", "每股经营现金流", "元"),
    ("eps", "每股收益EPS", "元"),
]


def format_stock_option(code: str, labels: dict[str, str]) -> str:
    name = labels.get(code, "")
    return f"{code} {name}" if name and name != code else code


def clean_industry_name(value) -> str:
    text = "" if value is None else str(value).strip()
    if text in {"", "未知", "未知行业", "nan", "None"}:
        return "未获取"
    return text


def format_financial_value(value, suffix: str) -> str:
    if value is None or pd.isna(value):
        return "未知"
    if suffix == "%":
        return f"{float(value):.2f}%"
    if suffix == "元":
        return f"{float(value):.4f}元"
    return str(value)


def make_financial_table(financial: dict) -> pd.DataFrame:
    rows = []
    for key, label, suffix in FINANCIAL_FIELDS:
        rows.append({"指标": label, "数值": format_financial_value(financial.get(key), suffix)})
    return pd.DataFrame(rows)


def friendly_data_warning(text: str) -> str:
    if "资金流数据源全部失败" in text:
        return text
    if "新闻数据源全部失败" in text:
        return text
    if "资金流接口不可用" in text:
        return "资金流接口暂未接通，已尝试备用源；稍后可点击“清空缓存并重新取数”。"
    if "新闻接口不可用" in text:
        return "新闻接口暂未接通，已尝试备用源。"
    if "公告接口不可用" in text:
        return "公告数据暂时不可用，不影响行情、财务和技术面分析。"
    if "财务指标接口不可用" in text:
        return "财务指标暂时不可用，当前主要依据行情和技术面。"
    if "ConnectionError" in text or "ProxyError" in text:
        return "部分数据源连接失败，通常是源站限流或网络波动。"
    if "ArrowInvalid" in text:
        return "部分数据源返回格式异常，已自动跳过该项。"
    return text


def render_data_warnings(warnings: list[str]) -> None:
    if not warnings:
        return
    messages = list(dict.fromkeys(friendly_data_warning(item) for item in warnings))
    with st.expander("数据源提示", expanded=False):
        for message in messages:
            st.write(f"- {message}")


def render_source_status(profile: dict) -> None:
    fund = profile.get("fund_flow_source") or "未接通"
    news = profile.get("news_source") or "未接通"
    notices = profile.get("notice_source") or "未接通"
    st.caption(f"资金流：{fund}；新闻：{news}；公告：{notices}")


def setup_auto_refresh(enabled: bool, interval_seconds: int) -> int:
    if not enabled:
        return 0
    st.sidebar.caption(f"自动刷新：每 {interval_seconds} 秒，最近刷新 {datetime.now():%H:%M:%S}")
    if st_autorefresh is not None:
        return int(st_autorefresh(interval=interval_seconds * 1000, key="market_auto_refresh"))

    # Fallback for environments without streamlit-autorefresh installed.
    components.html(
        f"""
        <script>
        setTimeout(function() {{
            window.parent.location.reload();
        }}, {interval_seconds * 1000});
        </script>
        """,
        height=0,
    )
    return int(time.time() // max(interval_seconds, 1))


def prepare_display_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        out[col] = out[col].map(format_display_cell)
    return out


def format_display_cell(value) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    return str(value)


def plot_price(df: pd.DataFrame, title: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Candlestick(
            x=df["date"],
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=df["close"],
            name="K线",
            increasing_line_color="#d62728",
            decreasing_line_color="#2ca02c",
        )
    )
    for col, color in [("ma5", "#f59e0b"), ("ma20", "#2563eb"), ("ma60", "#7c3aed")]:
        if col in df:
            fig.add_trace(
                go.Scatter(x=df["date"], y=df[col], mode="lines", name=col.upper(), line=dict(color=color, width=1.4))
            )
    fig.update_layout(
        title=title,
        template="plotly_white",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#ffffff",
        height=460,
        margin=dict(l=12, r=12, t=48, b=12),
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def plot_fund_flow(fund: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if isinstance(fund, pd.DataFrame) and not fund.empty and "主力净流入-净额" in fund:
        recent = fund.tail(40)
        fig.add_trace(
            go.Bar(
                x=recent["日期"],
                y=recent["主力净流入-净额"] / 1e8,
                name="主力净流入(亿元)",
                marker_color=["#dc2626" if v >= 0 else "#16a34a" for v in recent["主力净流入-净额"]],
            )
        )
    fig.update_layout(
        height=320,
        margin=dict(l=12, r=12, t=32, b=12),
        title="近40日主力资金流",
        template="plotly_white",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#ffffff",
    )
    return fig


def plot_backtest(equity: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if not equity.empty:
        fig.add_trace(go.Scatter(x=equity["date"], y=equity["equity"], mode="lines", name="策略权益"))
    fig.update_layout(
        height=320,
        margin=dict(l=12, r=12, t=32, b=12),
        title="策略权益曲线",
        template="plotly_white",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#ffffff",
    )
    return fig


def render_action_card(result: dict, compact: bool = False) -> None:
    color = result["color"]
    industry = clean_industry_name(result.get("industry"))
    meta = escape(f"{result['code']} {result['name']} / {industry}")
    action_label = escape(str(result["action_label"]))
    risk_level = escape(str(result["risk_level"]))
    confidence = escape(str(result["confidence"]))
    reason_html = ""
    if compact:
        reasons = [escape(str(item)) for item in result.get("reasons", [])[:2]]
        if reasons:
            reason_html = '<div class="action-card-reasons">' + "".join(f"<div>{item}</div>" for item in reasons) + "</div>"
    st.markdown(
        f"""
        <div class="action-card" style="border-left-color: {color};">
            <div class="action-card-meta">{meta}</div>
            <div class="action-card-title" style="color: {color};">{action_label}</div>
            <div class="action-card-score">综合分 {result['score']:.1f} / 风险 {risk_level} / 置信度 {confidence}</div>
            {reason_html}
        </div>
        """,
        unsafe_allow_html=True,
    )
    if not compact:
        st.write(result["summary"])


def holding_map_from_table(holdings: pd.DataFrame) -> dict:
    rows = {}
    for _, row in holdings.dropna(subset=["code"]).iterrows():
        codes = normalize_codes([row["code"]])
        if codes:
            rows[codes[0]] = {"cost": row.get("cost"), "weight": row.get("weight")}
    return rows


def scan_watchlist(
    codes: list[str],
    days: int,
    holdings: pd.DataFrame,
    realtime: bool,
    refresh_key: int = 0,
    labels: dict[str, str] | None = None,
) -> list[dict]:
    results = []
    labels = labels or {}
    holding_map = holding_map_from_table(holdings)
    total = max(len(codes), 1)
    progress = st.progress(0, text=f"准备扫描 {len(codes)} 只自选股...")
    for idx, code in enumerate(codes):
        display_name = labels.get(code, code)
        progress.progress(idx / total, text=f"正在拉取 {idx + 1}/{len(codes)}：{code} {display_name}")
        try:
            df, profile, source = load_stock(code, days, realtime, refresh_key, "scan")
        except Exception as exc:
            st.error(f"{code} 真实数据获取失败：{type(exc).__name__}。已跳过，未使用演示数据。")
            progress.progress((idx + 1) / total, text=f"已跳过 {idx + 1}/{len(codes)}：{code}")
            continue
        progress.progress(min((idx + 0.65) / total, 1.0), text=f"正在计算信号 {idx + 1}/{len(codes)}：{code} {display_name}")
        if labels.get(code):
            profile["name"] = labels[code]
        result = analyze_stock(code, df, profile, holding=holding_map.get(code))
        result["source"] = source
        results.append(result)
        progress.progress((idx + 1) / total, text=f"已完成 {idx + 1}/{len(codes)}：{code} {display_name}")
    progress.empty()
    return sorted(results, key=lambda x: (x["rank"], -x["score"]))


def make_summary_table(results: list[dict]) -> pd.DataFrame:
    rows = []
    for item in results:
        levels = item["levels"]
        metrics = item["metrics"]
        rows.append(
            {
                "代码": item["code"],
                "名称": item["name"],
                "行业": clean_industry_name(item.get("industry")),
                "操作": item["action_label"],
                "分数": round(item["score"], 1),
                "风险": item["risk_level"],
                "现价": round(item["last_close"], 2),
                "数据源": item.get("source", ""),
                "稳健买点": round(levels["conservative_entry"], 2),
                "突破买点": round(levels["breakout_entry"], 2),
                "止损位": round(levels["stop_loss"], 2),
                "5日涨跌幅": metrics.get("5日涨跌幅"),
                "10日涨跌幅": metrics.get("10日涨跌幅"),
                "20日涨跌幅": metrics.get("20日涨跌幅"),
                "RSI14": metrics.get("RSI14"),
                "核心原因": "；".join(item["reasons"][:2]),
            }
        )
    return pd.DataFrame(rows)


def render_news_table(df: pd.DataFrame, title: str) -> None:
    st.markdown(f"#### {title}")
    if not isinstance(df, pd.DataFrame) or df.empty:
        st.write("暂无可用数据，可能是接口暂时不可用。")
        return
    cols = [col for col in ["发布时间", "公告时间", "日期", "新闻标题", "公告标题", "标题", "文章来源", "新闻链接"] if col in df.columns]
    display = df[cols].head(10) if cols else df.head(10)
    st.dataframe(prepare_display_frame(display), use_container_width=True, hide_index=True)


def make_agent_table(review: dict) -> pd.DataFrame:
    rows = []
    for item in review["analysts"]:
        rows.append(
            {
                "角色": item["role"],
                "立场": item["stance"],
                "评分": item["score"],
                "主要支持": "；".join(item["evidence"]),
                "主要担忧": "；".join(item["concerns"]),
            }
        )
    return pd.DataFrame(rows)


def make_opportunity_table(rows: list[dict]) -> pd.DataFrame:
    columns = [
        "排名",
        "代码",
        "名称",
        "行业",
        "机会分",
        "初筛分",
        "五档评级",
        "操作",
        "风险",
        "现价",
        "稳健买点",
        "突破买点",
        "止损位",
        "PE",
        "PB",
        "成交额",
        "60日涨跌幅",
        "核心理由",
    ]
    if not rows:
        return pd.DataFrame(columns=columns)
    out = pd.DataFrame(rows)
    out.insert(0, "排名", range(1, len(out) + 1))
    return out[columns]


def _fmt_optional(value, digits: int = 2) -> str:
    try:
        if value is None or pd.isna(value):
            return "未知"
        return f"{float(value):.{digits}f}"
    except Exception:
        return "未知"


def _fmt_money(value) -> str:
    try:
        if value is None or pd.isna(value):
            return "未知"
        value = float(value)
    except Exception:
        return "未知"
    if abs(value) >= 1e8:
        return f"{value / 1e8:.2f}亿"
    if abs(value) >= 1e4:
        return f"{value / 1e4:.1f}万"
    return f"{value:.0f}"


def _opportunity_adjust(action: str) -> int:
    return {
        "buy": 10,
        "trial_buy": 6,
        "watch": 0,
        "hold": 0,
        "reduce": -10,
        "avoid": -18,
    }.get(action, 0)


def build_quality_reviews(
    candidates: pd.DataFrame | list[dict],
    deep_review_count: int,
    days: int,
    realtime: bool,
    refresh_key: int,
) -> list[dict]:
    records = candidates.to_dict("records") if isinstance(candidates, pd.DataFrame) else list(candidates)
    review_records = records[:deep_review_count]
    rows: list[dict] = []
    progress = st.progress(0, text="正在复核候选股...")
    for idx, candidate in enumerate(review_records):
        code = str(candidate["代码"])
        df, profile, source = load_stock(code, min(days, 260), realtime, refresh_key, "analysis")
        result = analyze_stock(code, df, profile)
        result["source"] = source
        review = build_agent_review(result)
        quick_score = float(candidate.get("初筛分", 50))
        opportunity_score = max(
            0,
            min(100, result["score"] * 0.62 + quick_score * 0.38 + _opportunity_adjust(result.get("action", ""))),
        )
        levels = result["levels"]
        rows.append(
            {
                "代码": code,
                "名称": result["name"],
                "行业": result.get("industry", "未知"),
                "机会分": round(opportunity_score, 1),
                "初筛分": round(quick_score, 1),
                "五档评级": review["rating"],
                "操作": result["action_label"],
                "风险": result["risk_level"],
                "现价": round(result["last_close"], 2),
                "稳健买点": round(levels["conservative_entry"], 2),
                "突破买点": round(levels["breakout_entry"], 2),
                "止损位": round(levels["stop_loss"], 2),
                "PE": _fmt_optional(candidate.get("市盈率-动态"), 1),
                "PB": _fmt_optional(candidate.get("市净率"), 1),
                "成交额": _fmt_money(candidate.get("成交额")),
                "60日涨跌幅": _fmt_optional(candidate.get("60日涨跌幅"), 1) + "%",
                "核心理由": "；".join((result["reasons"][:2] + [str(candidate.get("初筛理由", ""))])[:3]),
                "raw": result,
            }
        )
        progress.progress((idx + 1) / max(len(review_records), 1), text=f"已复核 {idx + 1}/{len(review_records)}")
    progress.empty()
    return sorted(rows, key=lambda item: item["机会分"], reverse=True)


with st.sidebar:
    if get_config_value("APP_PASSWORD") and st.session_state.get("authenticated"):
        if st.button("退出登录"):
            st.session_state.pop("authenticated", None)
            st.rerun()

    default_codes = (
        get_query_value("codes")
        or get_config_value("APP_DEFAULT_CODES")
        or "600519, 000001, 300750, 601318, 000858"
    )
    default_watchlist_codes = normalize_codes(default_codes)
    st.markdown(
        """
        <div class="side-panel">
            <div class="side-panel-head">
                <div>
                    <div class="side-panel-kicker">WATCHLIST</div>
                    <div class="side-panel-title">自选股池</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    stock_pool_seed = pd.DataFrame(
        [{"代码": code} for code in default_watchlist_codes]
        or [{"代码": ""}]
    )
    stock_pool_table = st.data_editor(
        stock_pool_seed,
        num_rows="dynamic",
        use_container_width=True,
        hide_index=True,
        key="watchlist_code_editor_v2",
        height=112,
        column_config={
            "代码": st.column_config.TextColumn("股票代码", width="medium", validate=r"^\d{0,6}$"),
        },
    )
    sidebar_codes = normalize_codes(stock_pool_table["代码"].tolist() if "代码" in stock_pool_table else [])
    code_text = ",".join(sidebar_codes)
    sidebar_labels = load_code_labels(tuple(sidebar_codes)) if sidebar_codes else {}
    synced_pool_table = pd.DataFrame(
        [{"代码": code, "名称": sidebar_labels.get(code, "未获取")} for code in sidebar_codes]
        or [{"代码": "", "名称": ""}]
    )
    st.dataframe(
        synced_pool_table,
        use_container_width=True,
        hide_index=True,
        height=min(112, 39 + 35 * max(len(synced_pool_table), 1)),
    )
    save_col, clear_col = st.columns(2)
    with save_col:
        if st.button("保存股票池", use_container_width=True):
            saved_codes = ",".join(sidebar_codes)
            if saved_codes:
                set_query_value("codes", saved_codes)
                st.success("已保存")
    with clear_col:
        if st.button("清除记忆", use_container_width=True):
            set_query_value("codes", "")
            st.session_state.pop("watchlist_editor", None)
            st.session_state.pop("watchlist_code_editor_v2", None)
            st.rerun()
    st.markdown('<div class="side-rule"></div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="side-panel">
            <div class="side-panel-head">
                <div>
                    <div class="side-panel-kicker">DATA WINDOW</div>
                    <div class="side-panel-title">周期和刷新</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    days = st.slider("分析周期", min_value=90, max_value=720, value=260, step=10)
    realtime = st.toggle("启用分钟线实时刷新", value=False, help="默认关闭会更快；盯盘时再打开，可把当天分钟线合入分析。")
    auto_refresh = st.toggle("自动刷新行情", value=False)
    refresh_seconds = 60
    if auto_refresh:
        refresh_seconds = st.selectbox("刷新间隔", [30, 60, 120, 300], index=1, format_func=lambda x: f"{x} 秒")
    if st.button("清空缓存并重新取数"):
        st.session_state["market_refresh_key"] = int(st.session_state.get("market_refresh_key", 0)) + 1
        clear_market_snapshot_file_cache()
        st.cache_data.clear()
        st.rerun()

    st.markdown('<div class="side-rule"></div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="side-panel">
            <div class="side-panel-head">
                <div>
                    <div class="side-panel-kicker">POSITION</div>
                    <div class="side-panel-title">持仓成本</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    with st.expander("录入持仓 / 成本 / 仓位", expanded=False):
        holdings = st.data_editor(
            pd.DataFrame([{"code": "", "cost": 0.0, "weight": 0.0}]),
            num_rows="dynamic",
            use_container_width=True,
            key="holdings_editor",
            column_config={
                "code": st.column_config.TextColumn("代码"),
                "cost": st.column_config.NumberColumn("成本价", min_value=0.0, step=0.01),
                "weight": st.column_config.NumberColumn("仓位%", min_value=0.0, max_value=100.0, step=1.0),
            },
        )
    sidebar_holding_rows = []
    if isinstance(holdings, pd.DataFrame) and not holdings.empty:
        for _, row in holdings.dropna(subset=["code"]).iterrows():
            normalized = normalize_codes([row.get("code", "")])
            if normalized:
                sidebar_holding_rows.append(
                    {
                        "code": normalized[0],
                        "cost": float(row.get("cost") or 0),
                        "weight": float(row.get("weight") or 0),
                    }
                )
    total_weight = sum(item["weight"] for item in sidebar_holding_rows)
    st.markdown(
        f"""
        <div class="side-metric-grid">
            <div class="side-metric">
                <div class="side-metric-label">持仓记录</div>
                <div class="side-metric-value">{len(sidebar_holding_rows)}</div>
            </div>
            <div class="side-metric">
                <div class="side-metric-label">仓位合计</div>
                <div class="side-metric-value">{total_weight:.0f}%</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

codes = normalize_codes(code_text)
if not codes:
    st.warning("请至少输入一个 6 位 A 股代码。")
    st.stop()
code_labels = load_code_labels(tuple(codes))
refresh_key = setup_auto_refresh(auto_refresh, refresh_seconds) if realtime else 0
render_app_header(len(codes), days, realtime, auto_refresh, refresh_seconds)

page_options = ["自选股扫描", "个股分析", "资金财务新闻", "简单回测", "板块轮动", "每日操作清单", "多智能体研判", "优质股票发现", "规则说明"]
requested_page = get_query_value("view")
current_page = requested_page if requested_page in page_options else page_options[0]
render_workflow_nav(current_page, codes)
render_research_scope(codes, code_labels, holdings)

if current_page == "自选股扫描":
    render_page_header("自选股扫描", "快速检查自选池里的趋势、风险和操作方向。", "WATCHLIST")
    render_section_note("当前为快速扫描模式，只拉行情和技术指标；ROE、资金流、新闻公告请进入“个股分析”或“资金财务新闻”查看。")
    results = scan_watchlist(codes, days, holdings, realtime, refresh_key, code_labels)
    brief = build_market_brief(results)
    cols = st.columns(4)
    cols[0].metric("扫描数量", len(results))
    cols[1].metric("买入/试探", brief["buy_count"])
    cols[2].metric("持有/观察", brief["watch_count"])
    cols[3].metric("减仓/回避", brief["risk_count"], f"均分 {brief['avg_score']:.1f}")

    st.markdown("### 信号明细")
    table_height = min(320, 76 + 36 * max(len(results), 2))
    st.dataframe(make_summary_table(results), use_container_width=True, hide_index=True, height=table_height)
    st.markdown("### 重点信号")
    visible_cards = results[:6]
    card_cols = st.columns(max(1, min(3, len(visible_cards))))
    for idx, item in enumerate(visible_cards):
        with card_cols[idx % len(card_cols)]:
            render_action_card(item, compact=True)

elif current_page == "个股分析":
    render_page_header("个股分析", "查看单只股票的技术结构、交易计划、风险提示和指标快照。", "DETAIL")
    selected_code = st.selectbox("选择股票", codes, format_func=lambda x: format_stock_option(x, code_labels))
    holding = holding_map_from_table(holdings).get(selected_code)
    df, profile, source = fetch_stock_with_progress(
        selected_code,
        days,
        realtime,
        refresh_key,
        "analysis",
        format_stock_option(selected_code, code_labels),
    )
    result = analyze_stock(selected_code, df, profile, holding=holding)
    result["source"] = source

    top_cols = st.columns([1, 2])
    with top_cols[0]:
        render_action_card(result)
        st.caption(f"数据源：{source}")
        render_source_status(profile)
        warnings = profile.get("data_warnings", [])
        render_data_warnings(warnings[:5])
    with top_cols[1]:
        st.plotly_chart(plot_price(result["frame"], f"{result['code']} {result['name']}"), use_container_width=True)

    levels = result["levels"]
    level_cols = st.columns(4)
    level_cols[0].metric("稳健买点", f"{levels['conservative_entry']:.2f}")
    level_cols[1].metric("突破买点", f"{levels['breakout_entry']:.2f}")
    level_cols[2].metric("止损位", f"{levels['stop_loss']:.2f}")
    level_cols[3].metric("止盈观察", f"{levels['take_profit_watch']:.2f}")

    st.markdown("#### 操作计划")
    st.write(result["operation_plan"])

    detail_cols = st.columns(3)
    with detail_cols[0]:
        st.markdown("##### 支持理由")
        for text in result["reasons"]:
            st.write(f"- {text}")
    with detail_cols[1]:
        st.markdown("##### 风险提示")
        for text in result["risks"]:
            st.write(f"- {text}")
    with detail_cols[2]:
        st.markdown("##### 指标快照")
        st.dataframe(pd.DataFrame(result["metrics"].items(), columns=["指标", "数值"]), hide_index=True, use_container_width=True)

    st.markdown("#### 操作报告")
    report = build_rule_report(result)
    st.markdown(report)

elif current_page == "资金财务新闻":
    render_page_header("资金、财务、新闻公告", "把资金流、财务质量、新闻和公告放在同一屏复核。", "DATA")
    selected_code = st.selectbox("选择股票查看明细", codes, key="detail_code", format_func=lambda x: format_stock_option(x, code_labels))
    df, profile, source = fetch_stock_with_progress(
        selected_code,
        days,
        realtime,
        refresh_key,
        "full",
        format_stock_option(selected_code, code_labels),
    )
    result = analyze_stock(selected_code, df, profile, holding=holding_map_from_table(holdings).get(selected_code))
    st.caption(f"数据源：{source}")
    render_source_status(profile)

    left, right = st.columns([1, 1])
    with left:
        st.plotly_chart(plot_fund_flow(profile.get("fund_flow")), use_container_width=True)
    with right:
        st.markdown("#### 财务指标")
        financial = profile.get("financial") or {}
        st.dataframe(make_financial_table(financial), use_container_width=True, hide_index=True)

    render_news_table(profile.get("news"), "新闻")
    render_news_table(profile.get("notices"), "公告")

elif current_page == "简单回测":
    render_page_header("简单回测", "用固定规则粗略验证均线趋势策略在历史区间里的表现。", "BACKTEST")
    selected_code = st.selectbox("选择股票回测", codes, key="bt_code", format_func=lambda x: format_stock_option(x, code_labels))
    df, profile, source = fetch_stock_with_progress(
        selected_code,
        days,
        False,
        0,
        "history",
        format_stock_option(selected_code, code_labels),
    )
    bt = run_ma_backtest(df)
    summary = bt["summary"]
    metric_cols = st.columns(6)
    if "样本不足" in summary:
        st.warning(summary["样本不足"])
    else:
        metric_cols[0].metric("策略收益", f"{summary['策略收益'] * 100:.1f}%")
        metric_cols[1].metric("买入持有", f"{summary['买入持有收益'] * 100:.1f}%")
        metric_cols[2].metric("最大回撤", f"{summary['最大回撤'] * 100:.1f}%")
        metric_cols[3].metric("交易次数", summary["交易次数"])
        metric_cols[4].metric("胜率", f"{summary['胜率'] * 100:.1f}%")
        metric_cols[5].metric("简化夏普", f"{summary['简化夏普']:.2f}")
        st.plotly_chart(plot_backtest(bt["equity"]), use_container_width=True)
        st.dataframe(bt["trades"], use_container_width=True, hide_index=True)
    st.caption("回测规则：站上20/60日均线且MACD为正时买入；跌破20日线、止损或过热止盈时卖出。")

elif current_page == "板块轮动":
    render_page_header("板块轮动", "观察行业强弱和自选池行业集中度。", "SECTOR")
    boards, board_source = fetch_boards_with_progress()
    st.caption(f"数据源：{board_source}")
    results = scan_watchlist(codes, days, holdings, realtime, refresh_key, code_labels)
    industry_table = (
        pd.DataFrame({"行业": [clean_industry_name(x.get("industry")) for x in results], "分数": [x["score"] for x in results]})
        .groupby("行业", as_index=False)
        .agg(股票数=("分数", "count"), 平均分=("分数", "mean"))
        .sort_values(["平均分", "股票数"], ascending=False)
    )
    board_col, pool_col = st.columns([1.45, 1])
    with board_col:
        st.markdown("### 市场板块列表")
        if boards.empty:
            st.warning("板块接口暂时不可用。你仍可以用右侧自选池行业暴露检查持仓集中度。")
        else:
            st.dataframe(boards.head(80), use_container_width=True, hide_index=True, height=420)
    with pool_col:
        st.markdown("### 自选池行业暴露")
        st.dataframe(industry_table, use_container_width=True, hide_index=True, height=210)
        st.markdown("### 自选池重点信号")
        for item in results[:4]:
            render_action_card(item, compact=True)

elif current_page == "每日操作清单":
    render_page_header("每日操作清单", "按买入、观察、减仓回避分组整理当天要看的动作。", date.today().isoformat())
    results = scan_watchlist(codes, days, holdings, realtime, refresh_key, code_labels)
    groups = {
        "可以买入/小仓试探": [x for x in results if x["action"] in {"buy", "trial_buy"}],
        "可以持有/继续观察": [x for x in results if x["action"] in {"hold", "watch"}],
        "建议减仓/卖出回避": [x for x in results if x["action"] in {"reduce", "avoid"}],
    }
    for title, items in groups.items():
        st.markdown(f"#### {title}")
        if not items:
            st.write("暂无。")
            continue
        for item in items:
            st.markdown(f"**{item['code']} {item['name']}：{item['action_label']}**")
            st.write(item["summary"])
            st.caption("；".join(item["reasons"][:3]))

elif current_page == "多智能体研判":
    render_page_header("多智能体研判", "从技术、资金、基本面、新闻和风控多个角度交叉复核。", "REVIEW")
    selected_code = st.selectbox("选择股票进行研判", codes, key="agent_code", format_func=lambda x: format_stock_option(x, code_labels))
    df, profile, source = fetch_stock_with_progress(
        selected_code,
        days,
        realtime,
        refresh_key,
        "full",
        format_stock_option(selected_code, code_labels),
    )
    result = analyze_stock(selected_code, df, profile, holding=holding_map_from_table(holdings).get(selected_code))
    result["source"] = source
    review = build_agent_review(result)

    top_cols = st.columns([1, 1, 1, 1])
    top_cols[0].metric("五档评级", review["rating"])
    top_cols[1].metric("执行动作", review["trader_action"])
    top_cols[2].metric("观点一致性", review["consensus"])
    top_cols[3].metric("规则综合分", f"{result['score']:.1f}")

    st.caption(f"数据源：{source}")
    render_source_status(profile)
    render_data_warnings(profile.get("data_warnings", [])[:5])

    left, right = st.columns([1, 1])
    with left:
        render_action_card(result)
    with right:
        st.markdown("#### 组合经理结论")
        st.write(review["research_manager"])
        st.markdown("#### 交易员执行单")
        st.write(f"**动作：**{review['trader_action']}")
        st.write(f"**入场：**{review['entry_text']}")
        st.write(f"**仓位：**{review['position_hint']}")

    st.markdown("#### 角色评分")
    st.dataframe(make_agent_table(review), use_container_width=True, hide_index=True)

    debate_cols = st.columns(2)
    with debate_cols[0]:
        st.markdown("#### 多方观点")
        for text in review["bull_case"]:
            st.write(f"- {text}")
    with debate_cols[1]:
        st.markdown("#### 空方观点")
        for text in review["bear_case"]:
            st.write(f"- {text}")

    control_cols = st.columns(2)
    with control_cols[0]:
        st.markdown("#### 风险控制")
        for text in review["risk_controls"]:
            st.write(f"- {text}")
    with control_cols[1]:
        st.markdown("#### 后续观察")
        for text in review["watch_items"]:
            st.write(f"- {text}")

    report_text = build_agent_markdown(result, review)
    st.download_button(
        "下载多智能体研判报告",
        data=report_text,
        file_name=f"{selected_code}_agent_review.md",
        mime="text/markdown",
        use_container_width=True,
    )

elif current_page == "优质股票发现":
    render_page_header("优质股票发现", "先用全市场快照粗筛，再对前排候选做技术、资金、财务、新闻和风险复核。", "DISCOVERY")
    render_section_note("这个页面只用于缩小研究范围，不构成买卖建议。真正操作仍要结合仓位、风险线和市场环境。")

    mode_options = list(MODE_LABELS.keys())
    discover_mode = st.selectbox(
        "筛选风格",
        mode_options,
        index=mode_options.index(st.session_state.get("quality_discover_mode", mode_options[0]))
        if st.session_state.get("quality_discover_mode") in mode_options
        else 0,
        format_func=lambda x: MODE_LABELS[x],
        key="quality_discover_mode",
        help="可以按不同研究方向切换，不是固定只找一种股票。",
    )
    discover_mode = st.session_state.get("quality_discover_mode", discover_mode)
    st.info(f"{MODE_LABELS[discover_mode]}：{MODE_DESCRIPTIONS[discover_mode]}")

    with st.form("quality_discovery_form"):
        control_cols = st.columns([1, 1, 1])
        with control_cols[0]:
            universe_limit = st.slider("初筛候选数", min_value=20, max_value=120, value=60, step=10)
        with control_cols[1]:
            deep_review_count = st.slider(
                "深度复核数",
                min_value=3,
                max_value=15,
                value=5,
                step=1,
                help="深度复核会逐只拉K线、资金、财务和新闻，数量越大越慢。",
            )
        with control_cols[2]:
            show_count = st.slider("展示数量", min_value=5, max_value=20, value=10, step=1)

        scan_mode = st.radio(
            "扫描方式",
            ["快速初筛", "深度复核"],
            index=0,
            horizontal=True,
            help="快速初筛只做全市场快照打分；深度复核会逐只补充K线、资金、财务和新闻，结果更细但更慢。",
        )

        with st.expander("我是怎么筛选这些候选股的", expanded=False):
            st.markdown(
                """
                这不是直接“预测哪只一定涨”，而是先把全市场缩小成值得研究的候选池：

                1. 数据源优先级：东方财富全A实时快照直连，其次 AKShare 东方财富全市场快照，再用 AKShare 新浪全A实时快照；如果真实数据源都失败，就停止筛选，不使用演示候选池。
                2. 基础排除：先排除 ST、退市、新股标记、成交过低、极端估值、涨跌停附近和换手异常的股票。
                3. 初筛打分：综合估值 PE/PB、流动性、60日趋势、当日强度、换手/量比/振幅、主力净流入和短期风险。
                4. 风格权重：稳健优质更均衡，趋势增强更看动量，低估修复更看估值，资金关注更看成交和资金，突破/超跌用于找特定形态。
                5. 深度复核：只对初筛前排做现有个股分析，再用技术面、资金面、基本面、新闻公告和风险经理多角色复核。
                """
            )

        run_discovery = st.form_submit_button("开始扫描优质候选", type="primary", use_container_width=True)
    if run_discovery:
        market_refresh_key = int(st.session_state.get("market_refresh_key", 0))
        snapshot, snapshot_source, snapshot_warnings = fetch_market_snapshot_with_progress(market_refresh_key)
        candidates = screen_market_candidates(snapshot, mode=discover_mode, limit=universe_limit)
        if snapshot_warnings:
            render_data_warnings(snapshot_warnings)
        if "不可用" in snapshot_source:
            st.error("当前没有连上真实全A市场，系统已停止筛选，未使用演示候选池。请点左侧“清空缓存并重新取数”，或稍后重试。")
        elif "新浪" in snapshot_source:
            st.info("当前使用新浪全A备用源，能覆盖真实全市场实时行情，但 PE/PB、60日涨跌幅等字段较少；系统会先用价格和成交额粗筛，再在深度复核时补充个股技术、资金、财务和新闻数据。")
        st.caption(f"候选池数据源：{snapshot_source}；快照股票数 {len(snapshot)}；初筛展示 {len(candidates)} 只")
        if candidates.empty:
            st.warning("当前没有筛出候选股。可以换一个筛选风格，或稍后再重新取数。")
        else:
            st.session_state["quality_fast_candidates"] = candidates.head(show_count).to_dict("records")
            st.session_state["quality_candidates"] = []
            if scan_mode == "深度复核":
                rows = build_quality_reviews(candidates, deep_review_count, days, realtime, refresh_key)[:show_count]
                st.session_state["quality_candidates"] = rows
            else:
                st.success("已完成快速初筛。需要更细的买点、风险和多角色观点时，再点击下方“对这些候选做深度复核”。")

    quality_rows = st.session_state.get("quality_candidates", [])
    fast_rows = st.session_state.get("quality_fast_candidates", [])
    if fast_rows and not quality_rows:
        st.markdown("#### 快速初筛候选")
        fast_table = []
        for rank, row in enumerate(fast_rows, start=1):
            fast_table.append(
                {
                    "排名": rank,
                    "代码": row.get("代码"),
                    "名称": row.get("名称"),
                    "初筛分": row.get("初筛分"),
                    "现价": _fmt_optional(row.get("最新价"), 2),
                    "PE": _fmt_optional(row.get("市盈率-动态"), 1),
                    "PB": _fmt_optional(row.get("市净率"), 1),
                    "成交额": _fmt_money(row.get("成交额")),
                    "60日涨跌幅": _fmt_optional(row.get("60日涨跌幅"), 1) + "%",
                    "初筛理由": row.get("初筛理由", ""),
                }
            )
        st.dataframe(pd.DataFrame(fast_table), use_container_width=True, hide_index=True)
        fast_cols = st.columns([1, 1, 2])
        with fast_cols[0]:
            if st.button("对这些候选做深度复核", use_container_width=True):
                rows = build_quality_reviews(fast_rows, deep_review_count, days, realtime, refresh_key)[:show_count]
                st.session_state["quality_candidates"] = rows
                st.rerun()
        with fast_cols[1]:
            if st.button("把初筛候选加入自选股", use_container_width=True):
                merged_codes = normalize_codes(code_text) + [str(row["代码"]) for row in fast_rows]
                merged_text = ",".join(dict.fromkeys(merged_codes))
                st.session_state.pop("watchlist_code_editor_v2", None)
                set_query_value("codes", merged_text)
                st.success("已加入左侧股票池，并写入当前网址。")
                st.rerun()
        with fast_cols[2]:
            st.caption("快速初筛主要用于缩小范围，深度复核才会补充买点、止损、新闻和多角色观点。")

    if quality_rows:
        st.markdown("#### 优质候选榜")
        table = make_opportunity_table(quality_rows)
        st.dataframe(table, use_container_width=True, hide_index=True)

        add_cols = st.columns([1, 2])
        with add_cols[0]:
            if st.button("把候选加入自选股", use_container_width=True):
                merged_codes = normalize_codes(code_text) + [row["代码"] for row in quality_rows]
                merged_text = ",".join(dict.fromkeys(merged_codes))
                st.session_state.pop("watchlist_code_editor_v2", None)
                set_query_value("codes", merged_text)
                st.success("已加入左侧股票池，并写入当前网址。")
                st.rerun()
        with add_cols[1]:
            st.caption("机会分越高，只代表更值得进一步研究；真正买入仍要看你的仓位、风险线和市场环境。")

        selected_idx = st.selectbox(
            "查看候选详情",
            list(range(len(quality_rows))),
            format_func=lambda i: f"{quality_rows[i]['代码']} {quality_rows[i]['名称']} - {quality_rows[i]['五档评级']} / {quality_rows[i]['机会分']}",
        )
        selected_candidate = quality_rows[selected_idx]
        detail = selected_candidate["raw"]
        detail_review = build_agent_review(detail)
        dcols = st.columns([1, 1])
        with dcols[0]:
            render_action_card(detail)
            st.write(detail["operation_plan"])
        with dcols[1]:
            st.markdown("#### 为什么入选")
            for text in detail["reasons"][:5]:
                st.write(f"- {text}")
            st.markdown("#### 主要风险")
            for text in detail["risks"][:5]:
                st.write(f"- {text}")
        st.markdown("#### 多角色复核")
        st.dataframe(make_agent_table(detail_review), use_container_width=True, hide_index=True)
    else:
        st.info("点击“开始扫描优质候选”，系统会从市场快照里筛出一批更值得进一步研究的股票。")

elif current_page == "规则说明":
    render_page_header("规则说明", "说明当前系统如何打分、如何判断风险，以及哪些数据会影响结论。", "RULES")
    st.write(
        """
        这一版不再只看技术指标，而是把数据拆成几层独立判断：

        1. 实时行情：新浪日线 + 分钟线，尽量把当天最新价格合进K线。
        2. 技术面：均线、MACD、RSI、量能、波动率、支撑压力。
        3. 资金面：个股主力资金近1日、5日、20日净流入。
        4. 基本面：最新财报的ROE、营收增长、净利润增长、资产负债率、现金流。
        5. 消息面：新闻和公告关键词，尤其关注减持、处罚、退市、亏损、下滑等风险词。
        6. 回测：验证一条简单趋势规则过去是否有效，不把历史收益当成未来承诺。
        7. 多智能体研判：参考 TradingAgents 的角色拆分方式，把技术面、资金面、基本面、消息面和风险控制分别打分，再由组合经理汇总成五档评级。
        8. 优质股票发现：参考 Qlib/FinRL 的“先筛信号、再做复核”的量化流程，先按流动性、估值、趋势和风险过滤市场，再对前排候选做详细分析。

        操作报告由本地规则生成，不调用外部接口。
        """
    )
