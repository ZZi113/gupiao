from __future__ import annotations

import os
from datetime import date

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.ashare_advisor.backtest import run_ma_backtest
from src.ashare_advisor.data import DataProvider, normalize_codes
from src.ashare_advisor.reporting import build_rule_report, generate_llm_report
from src.ashare_advisor.rules import analyze_stock, build_market_brief


st.set_page_config(
    page_title="A股个人投资操作助手",
    page_icon="📈",
    layout="wide",
)


def get_config_value(name: str, default: str = "") -> str:
    env_value = os.getenv(name)
    if env_value:
        return env_value
    try:
        value = st.secrets.get(name, default)
    except Exception:
        return default
    return str(value) if value is not None else default


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


@st.cache_data(show_spinner=False, ttl=60 * 5)
def load_stock(code: str, days: int, realtime: bool) -> tuple[pd.DataFrame, dict, str]:
    provider = DataProvider()
    return provider.load_stock(code, days=days, realtime=realtime)


@st.cache_data(show_spinner=False, ttl=60 * 60)
def load_boards() -> tuple[pd.DataFrame, str]:
    provider = DataProvider()
    return provider.load_industry_boards()


@st.cache_data(show_spinner=False, ttl=60 * 60)
def load_code_labels(codes: tuple[str, ...]) -> dict[str, str]:
    provider = DataProvider()
    return provider.load_code_name_map(codes)


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
    fig.update_layout(height=320, margin=dict(l=12, r=12, t=32, b=12), title="近40日主力资金流")
    return fig


def plot_backtest(equity: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if not equity.empty:
        fig.add_trace(go.Scatter(x=equity["date"], y=equity["equity"], mode="lines", name="策略权益"))
    fig.update_layout(height=320, margin=dict(l=12, r=12, t=32, b=12), title="策略权益曲线")
    return fig


def render_action_card(result: dict, compact: bool = False) -> None:
    color = result["color"]
    st.markdown(
        f"""
        <div style="border-left: 6px solid {color}; padding: 12px 14px; background: #f8fafc; border-radius: 6px;">
            <div style="font-size: 14px; color: #64748b;">{result['code']} {result['name']} / {result.get('industry', '未知')}</div>
            <div style="font-size: 24px; font-weight: 700; color: {color};">{result['action_label']}</div>
            <div style="font-size: 14px; color: #334155;">综合分 {result['score']:.1f} / 风险 {result['risk_level']} / 置信度 {result['confidence']}</div>
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


def scan_watchlist(codes: list[str], days: int, holdings: pd.DataFrame, realtime: bool) -> list[dict]:
    results = []
    holding_map = holding_map_from_table(holdings)
    progress = st.progress(0, text="正在扫描自选股...")
    for idx, code in enumerate(codes):
        df, profile, source = load_stock(code, days, realtime)
        result = analyze_stock(code, df, profile, holding=holding_map.get(code))
        result["source"] = source
        results.append(result)
        progress.progress((idx + 1) / max(len(codes), 1), text=f"已完成 {idx + 1}/{len(codes)}")
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
                "行业": item.get("industry", "未知"),
                "操作": item["action_label"],
                "分数": round(item["score"], 1),
                "风险": item["risk_level"],
                "现价": round(item["last_close"], 2),
                "稳健买点": round(levels["conservative_entry"], 2),
                "突破买点": round(levels["breakout_entry"], 2),
                "止损位": round(levels["stop_loss"], 2),
                "ROE": metrics.get("ROE"),
                "净利润增长": metrics.get("净利润增长"),
                "5日主力": metrics.get("5日主力净流入"),
                "更新时间": metrics.get("实时更新时间"),
                "核心原因": "；".join(item["reasons"][:2]),
                "数据源": item.get("source", ""),
            }
        )
    return pd.DataFrame(rows)


def render_news_table(df: pd.DataFrame, title: str) -> None:
    st.markdown(f"#### {title}")
    if not isinstance(df, pd.DataFrame) or df.empty:
        st.write("暂无可用数据，可能是接口暂时不可用。")
        return
    cols = [col for col in ["发布时间", "公告时间", "日期", "新闻标题", "公告标题", "标题", "文章来源", "新闻链接"] if col in df.columns]
    st.dataframe(df[cols].head(10) if cols else df.head(10), use_container_width=True, hide_index=True)


st.title("A股个人投资操作助手")
st.caption("实时行情、资金流、财务、新闻公告和简单回测组合判断。仅供研究，不构成投资建议。")

with st.sidebar:
    st.header("股票池")
    if get_config_value("APP_PASSWORD") and st.session_state.get("authenticated"):
        if st.button("退出登录"):
            st.session_state.pop("authenticated", None)
            st.rerun()

    default_codes = "600519, 000001, 300750, 601318, 000858"
    code_text = st.text_area("自选股代码", value=default_codes, height=110)
    days = st.slider("分析周期", min_value=90, max_value=720, value=260, step=10)
    realtime = st.toggle("启用分钟线实时刷新", value=True)
    if st.button("清空缓存并重新取数"):
        st.cache_data.clear()
        st.rerun()

    st.subheader("持仓，可选")
    holdings = st.data_editor(
        pd.DataFrame([{"code": "", "cost": 0.0, "weight": 0.0}]),
        num_rows="dynamic",
        use_container_width=True,
        column_config={
            "code": st.column_config.TextColumn("代码"),
            "cost": st.column_config.NumberColumn("成本价", min_value=0.0, step=0.01),
            "weight": st.column_config.NumberColumn("仓位%", min_value=0.0, max_value=100.0, step=1.0),
        },
    )

    st.subheader("大模型，可选")
    use_llm = st.toggle("生成大模型报告", value=False)
    default_llm_base_url = get_config_value("LLM_BASE_URL")
    default_llm_model = get_config_value("LLM_MODEL", "gpt-4o-mini")
    default_llm_api_key = get_config_value("LLM_API_KEY")
    llm_base_url = st.text_input("OpenAI兼容接口地址", value=default_llm_base_url, placeholder="例如 https://api.openai.com/v1")
    llm_model = st.text_input("模型名", value=default_llm_model)
    llm_api_key = st.text_input(
        "API Key",
        value="",
        type="password",
        placeholder="留空则使用云端密钥" if default_llm_api_key else "",
    )

codes = normalize_codes(code_text)
if not codes:
    st.warning("请至少输入一个 6 位 A 股代码。")
    st.stop()
code_labels = load_code_labels(tuple(codes))

tabs = st.tabs(["自选股扫描", "个股分析", "资金财务新闻", "简单回测", "板块轮动", "每日操作清单", "规则说明"])

with tabs[0]:
    st.subheader("自选股扫描")
    results = scan_watchlist(codes, days, holdings, realtime)
    brief = build_market_brief(results)
    cols = st.columns(4)
    cols[0].metric("扫描数量", len(results))
    cols[1].metric("买入/试探", brief["buy_count"])
    cols[2].metric("持有/观察", brief["watch_count"])
    cols[3].metric("减仓/回避", brief["risk_count"], f"均分 {brief['avg_score']:.1f}")

    st.dataframe(make_summary_table(results), use_container_width=True, hide_index=True)
    st.markdown("#### 重点卡片")
    card_cols = st.columns(3)
    for idx, item in enumerate(results[:6]):
        with card_cols[idx % 3]:
            render_action_card(item, compact=True)
            st.write("；".join(item["reasons"][:3]))

with tabs[1]:
    st.subheader("个股分析")
    selected_code = st.selectbox("选择股票", codes, format_func=lambda x: format_stock_option(x, code_labels))
    holding = holding_map_from_table(holdings).get(selected_code)
    df, profile, source = load_stock(selected_code, days, realtime)
    result = analyze_stock(selected_code, df, profile, holding=holding)
    result["source"] = source

    top_cols = st.columns([1, 2])
    with top_cols[0]:
        render_action_card(result)
        st.caption(f"数据源：{source}")
        warnings = profile.get("data_warnings", [])
        if warnings:
            st.warning("；".join(warnings[:3]))
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
    if use_llm:
        report = generate_llm_report(
            result,
            api_key=llm_api_key or default_llm_api_key,
            base_url=llm_base_url or default_llm_base_url,
            model=llm_model or default_llm_model,
        )
    else:
        report = build_rule_report(result)
    st.markdown(report)

with tabs[2]:
    st.subheader("资金、财务、新闻公告")
    selected_code = st.selectbox("选择股票查看明细", codes, key="detail_code", format_func=lambda x: format_stock_option(x, code_labels))
    df, profile, source = load_stock(selected_code, days, realtime)
    result = analyze_stock(selected_code, df, profile, holding=holding_map_from_table(holdings).get(selected_code))
    st.caption(f"数据源：{source}")

    left, right = st.columns([1, 1])
    with left:
        st.plotly_chart(plot_fund_flow(profile.get("fund_flow")), use_container_width=True)
    with right:
        st.markdown("#### 财务指标")
        financial = profile.get("financial") or {}
        st.dataframe(make_financial_table(financial), use_container_width=True, hide_index=True)

    render_news_table(profile.get("news"), "新闻")
    render_news_table(profile.get("notices"), "公告")

with tabs[3]:
    st.subheader("简单回测")
    selected_code = st.selectbox("选择股票回测", codes, key="bt_code", format_func=lambda x: format_stock_option(x, code_labels))
    df, profile, source = load_stock(selected_code, days, False)
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

with tabs[4]:
    st.subheader("板块轮动")
    boards, board_source = load_boards()
    st.caption(f"数据源：{board_source}")
    if boards.empty:
        st.warning("板块接口暂时不可用。你仍可以用自选股扫描中的行业列做持仓行业集中度检查。")
    else:
        st.dataframe(boards.head(100), use_container_width=True, hide_index=True)

    st.markdown("#### 自选股行业分布")
    results = scan_watchlist(codes, days, holdings, realtime)
    industry_table = (
        pd.DataFrame({"行业": [x.get("industry", "未知") for x in results], "分数": [x["score"] for x in results]})
        .groupby("行业", as_index=False)
        .agg(股票数=("分数", "count"), 平均分=("分数", "mean"))
        .sort_values(["平均分", "股票数"], ascending=False)
    )
    st.dataframe(industry_table, use_container_width=True, hide_index=True)

with tabs[5]:
    st.subheader(f"每日操作清单 - {date.today().isoformat()}")
    results = scan_watchlist(codes, days, holdings, realtime)
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

with tabs[6]:
    st.subheader("规则说明")
    st.write(
        """
        这一版不再只看技术指标，而是把数据拆成几层独立判断：

        1. 实时行情：新浪日线 + 分钟线，尽量把当天最新价格合进K线。
        2. 技术面：均线、MACD、RSI、量能、波动率、支撑压力。
        3. 资金面：个股主力资金近1日、5日、20日净流入。
        4. 基本面：最新财报的ROE、营收增长、净利润增长、资产负债率、现金流。
        5. 消息面：新闻和公告关键词，尤其关注减持、处罚、退市、亏损、下滑等风险词。
        6. 回测：验证一条简单趋势规则过去是否有效，不把历史收益当成未来承诺。

        大模型报告只负责把结构化数据改写成更自然的操作报告，不负责凭空预测涨跌。
        """
    )
