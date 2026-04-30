from __future__ import annotations

import json
import re
import time
from datetime import date, datetime, timedelta
from typing import Iterable

import pandas as pd
import requests

from .indicators import add_indicators


_CODE_NAME_CACHE: pd.DataFrame | None = None
_CODE_LABEL_CACHE: dict[str, str] = {}


def normalize_codes(text_or_codes: str | Iterable[str]) -> list[str]:
    if isinstance(text_or_codes, str):
        raw = re.split(r"[\s,，;；]+", text_or_codes.strip())
    else:
        raw = list(text_or_codes)
    codes = []
    for item in raw:
        digits = re.sub(r"\D", "", str(item))
        if len(digits) >= 6:
            codes.append(digits[-6:])
    return list(dict.fromkeys(codes))


def market_prefix(code: str) -> str:
    if code.startswith(("6", "9")):
        return "sh"
    if code.startswith(("4", "8")):
        return "bj"
    return "sz"


def sina_symbol(code: str) -> str:
    return f"{market_prefix(code)}{code}"


def _num(value, default=None):
    try:
        if value in ("-", "--", "", None):
            return default
        out = pd.to_numeric(value, errors="coerce")
        if pd.isna(out):
            return default
        return float(out)
    except Exception:
        return default


def _safe_date(value):
    try:
        return pd.to_datetime(value)
    except Exception:
        return pd.NaT


def _request_json(url: str, params: dict, headers: dict | None = None, timeout: int = 12) -> dict:
    last_error: Exception | None = None
    for trust_env in (True, False):
        try:
            session = requests.Session()
            session.trust_env = trust_env
            response = session.get(url, params=params, headers=headers or {}, timeout=timeout)
            response.raise_for_status()
            return response.json()
        except Exception as exc:
            last_error = exc
    raise last_error or RuntimeError("request failed")


def _request_text(url: str, params: dict, headers: dict | None = None, timeout: int = 12) -> str:
    last_error: Exception | None = None
    for trust_env in (True, False):
        try:
            session = requests.Session()
            session.trust_env = trust_env
            response = session.get(url, params=params, headers=headers or {}, timeout=timeout)
            response.raise_for_status()
            return response.text
        except Exception as exc:
            last_error = exc
    raise last_error or RuntimeError("request failed")


class DataProvider:
    def __init__(self) -> None:
        try:
            import akshare as ak  # type: ignore
        except Exception:
            ak = None
        self.ak = ak

    def load_stock(
        self,
        code: str,
        days: int = 260,
        realtime: bool = True,
        detail_level: str = "full",
    ) -> tuple[pd.DataFrame, dict, str]:
        code = normalize_codes([code])[0]
        profile = {
            "code": code,
            "name": code,
            "industry": "",
            "pe": None,
            "pb": None,
            "roe": None,
            "debt_ratio": None,
            "data_warnings": [],
            "data_notes": [],
            "realtime": {},
            "financial": {},
            "fund_flow": pd.DataFrame(),
            "fund_flow_source": "",
            "news": pd.DataFrame(),
            "news_source": "",
            "notices": pd.DataFrame(),
            "notice_source": "",
        }

        if self.ak is None:
            raise RuntimeError("未安装 AKShare，无法获取真实行情")

        source_parts: list[str] = []
        df = self._load_real_history(code, days, profile, source_parts)
        if detail_level == "scan":
            self._enrich_profile(code, profile)
            source_parts.append("行情 + 个股资料")
        elif detail_level == "history":
            source_parts.append("历史行情")
        elif detail_level == "analysis":
            self._enrich_profile(code, profile)
            self._enrich_financials(code, profile)
            self._enrich_fund_flow(code, profile)
            source_parts.append("快速分析")
        else:
            self._enrich_profile(code, profile)
            self._enrich_financials(code, profile)
            self._enrich_fund_flow(code, profile)
            self._enrich_news(code, profile)
            self._enrich_notices(code, profile)

        if realtime:
            minute_bar = self._load_realtime_minute_bar(code, profile)
            if minute_bar is not None:
                df = self._merge_realtime_bar(df, minute_bar)
                source_parts.append("新浪分钟线")

        if df is None or df.empty:
            raise RuntimeError("真实行情接口不可用，未使用演示行情替代")

        final_df = df.tail(days).reset_index(drop=True)
        latest_date = pd.to_datetime(final_df["date"], errors="coerce").max() if "date" in final_df else pd.NaT
        profile["history_rows"] = int(len(final_df))
        profile["history_latest_date"] = "" if pd.isna(latest_date) else latest_date.strftime("%Y-%m-%d")
        profile["realtime_requested"] = bool(realtime)
        if realtime and not profile.get("realtime"):
            profile["data_notes"].append("已请求分钟线实时数据，但当前接口未返回可合并的当日分钟线")
        elif not realtime:
            profile["data_notes"].append("当前使用日线快照；如需盯盘同步，请开启分钟线实时刷新")

        return add_indicators(final_df), profile, " + ".join(dict.fromkeys(source_parts))

    def _load_real_history(self, code: str, days: int, profile: dict, source_parts: list[str]) -> pd.DataFrame | None:
        start = (date.today() - timedelta(days=days * 2)).strftime("%Y%m%d")
        end = date.today().strftime("%Y%m%d")
        loaders = [
            ("新浪日线", lambda: self._load_from_sina_daily(code, start, end)),
            ("腾讯日线", lambda: self._load_from_tx(code, start, end)),
            ("东方财富日线", lambda: self._load_from_eastmoney(code, start, end)),
        ]
        last_error = None
        for name, loader in loaders:
            try:
                df = loader()
                if df is not None and not df.empty:
                    source_parts.append(name)
                    return df.tail(days).reset_index(drop=True)
            except Exception as exc:
                last_error = exc
        if last_error is not None:
            profile["data_warnings"].append(f"历史行情接口失败：{type(last_error).__name__}")
        return None

    def _load_from_sina_daily(self, code: str, start: str, end: str) -> pd.DataFrame:
        raw = self.ak.stock_zh_a_daily(symbol=sina_symbol(code), start_date=start, end_date=end, adjust="qfq")
        if raw.empty:
            raise ValueError("empty sina daily")
        df = raw.rename(columns={"date": "date"}).copy()
        df["date"] = pd.to_datetime(df["date"])
        keep = ["date", "open", "high", "low", "close", "volume", "amount", "turnover"]
        for col in keep:
            if col not in df:
                df[col] = 0.0 if col not in {"date"} else pd.NaT
        return df[keep].dropna(subset=["date", "open", "high", "low", "close"])

    def _load_from_tx(self, code: str, start: str, end: str) -> pd.DataFrame:
        raw = self.ak.stock_zh_a_hist_tx(symbol=sina_symbol(code), start_date=start, end_date=end, adjust="qfq")
        if raw.empty:
            raise ValueError("empty tx daily")
        df = raw.rename(columns={"amount": "volume"}).copy()
        df["date"] = pd.to_datetime(df["date"])
        df["amount"] = df["volume"] * df["close"]
        df["turnover"] = 0.0
        return df[["date", "open", "high", "low", "close", "volume", "amount", "turnover"]]

    def _load_from_eastmoney(self, code: str, start: str, end: str) -> pd.DataFrame:
        raw = self.ak.stock_zh_a_hist(symbol=code, period="daily", start_date=start, end_date=end, adjust="qfq")
        if raw.empty:
            raise ValueError("empty eastmoney daily")
        column_map = {
            "日期": "date",
            "开盘": "open",
            "收盘": "close",
            "最高": "high",
            "最低": "low",
            "成交量": "volume",
            "成交额": "amount",
            "换手率": "turnover",
        }
        df = raw.rename(columns=column_map)
        for col in ["open", "high", "low", "close", "volume", "amount", "turnover"]:
            if col in df:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        df["date"] = pd.to_datetime(df["date"])
        if "turnover" not in df:
            df["turnover"] = 0.0
        return df[["date", "open", "high", "low", "close", "volume", "amount", "turnover"]].dropna()

    def _load_realtime_minute_bar(self, code: str, profile: dict) -> dict | None:
        try:
            raw = self.ak.stock_zh_a_minute(symbol=sina_symbol(code), period="1", adjust="")
            if raw.empty:
                return None
            df = raw.rename(columns={"day": "datetime"}).copy()
            df["datetime"] = pd.to_datetime(df["datetime"])
            latest_day = df["datetime"].dt.date.max()
            today_rows = df[df["datetime"].dt.date == latest_day].copy()
            if today_rows.empty:
                return None
            latest = today_rows.iloc[-1]
            volume_sum = pd.to_numeric(today_rows.get("volume"), errors="coerce").fillna(0).sum()
            amount_sum = pd.to_numeric(today_rows.get("amount"), errors="coerce").fillna(0).sum()
            profile["realtime"] = {
                "price": _num(latest.get("close")),
                "time": str(latest.get("datetime")),
                "volume": float(volume_sum),
                "amount": float(amount_sum),
            }
            return {
                "date": pd.to_datetime(latest_day),
                "open": _num(today_rows.iloc[0].get("open")),
                "high": _num(today_rows["high"].max()),
                "low": _num(today_rows["low"].min()),
                "close": _num(latest.get("close")),
                "volume": float(volume_sum),
                "amount": float(amount_sum),
                "turnover": 0.0,
            }
        except Exception as exc:
            profile["data_warnings"].append(f"实时分钟线不可用：{type(exc).__name__}")
            return None

    def _merge_realtime_bar(self, df: pd.DataFrame, bar: dict) -> pd.DataFrame:
        out = df.copy()
        bar_date = pd.to_datetime(bar["date"]).normalize()
        out["date_norm"] = pd.to_datetime(out["date"]).dt.normalize()
        if bar_date in set(out["date_norm"]):
            idx = out.index[out["date_norm"] == bar_date][-1]
            for key in ["open", "high", "low", "close", "volume", "amount", "turnover"]:
                out.loc[idx, key] = bar[key]
        else:
            out = pd.concat([out.drop(columns=["date_norm"]), pd.DataFrame([bar])], ignore_index=True)
            return out.sort_values("date")
        return out.drop(columns=["date_norm"]).sort_values("date")

    def _enrich_profile(self, code: str, profile: dict) -> None:
        primary_error = None
        try:
            info = self.ak.stock_individual_info_em(symbol=code)
            values = dict(zip(info["item"].astype(str), info["value"]))
            profile["name"] = str(values.get("股票简称", profile.get("name", code)))
            profile["industry"] = str(values.get("行业", profile.get("industry", "未知")))
            profile["total_market_value"] = _num(values.get("总市值"))
            profile["free_market_value"] = _num(values.get("流通市值"))
        except Exception as exc:
            primary_error = type(exc).__name__
        self._enrich_profile_from_cninfo(code, profile)
        self._enrich_name_from_code_table(code, profile)
        unresolved_name = profile.get("name") in {None, "", code, f"股票{code}"}
        unresolved_industry = profile.get("industry") in {None, "", "未知", "未知行业"}
        if primary_error and (unresolved_name or unresolved_industry):
            profile["data_warnings"].append(f"个股资料接口不可用：{primary_error}")

    def _enrich_profile_from_cninfo(self, code: str, profile: dict) -> None:
        needs_name = profile.get("name") in {None, "", code, f"股票{code}"}
        needs_industry = profile.get("industry") in {None, "", "未知", "未知行业"}
        if not needs_name and not needs_industry:
            return
        try:
            raw = self.ak.stock_profile_cninfo(symbol=code)
            if raw.empty:
                return
            row = raw.iloc[0]
            name = row.get("A股简称")
            industry = row.get("所属行业")
            company = row.get("公司名称")
            if needs_name and pd.notna(name):
                profile["name"] = str(name)
            if needs_industry and pd.notna(industry):
                profile["industry"] = str(industry)
            if pd.notna(company):
                profile["company_name"] = str(company)
        except Exception as exc:
            profile["data_warnings"].append(f"巨潮资料接口不可用：{type(exc).__name__}")

    def _enrich_name_from_code_table(self, code: str, profile: dict) -> None:
        if profile.get("name") not in {None, "", code, f"股票{code}"}:
            return
        try:
            table = self._load_code_name_table()
            if table.empty:
                return
            code_col, name_col = table.columns[:2]
            row = table[table[code_col].astype(str).str.zfill(6).eq(code)]
            if not row.empty:
                profile["name"] = str(row.iloc[0][name_col])
        except Exception as exc:
            profile["data_warnings"].append(f"代码名称表不可用：{type(exc).__name__}")

    def _load_code_name_table(self) -> pd.DataFrame:
        global _CODE_NAME_CACHE
        if _CODE_NAME_CACHE is None:
            raw = self.ak.stock_info_a_code_name()
            _CODE_NAME_CACHE = raw.copy()
        return _CODE_NAME_CACHE

    def load_code_name_map(self, codes: Iterable[str]) -> dict[str, str]:
        normalized = normalize_codes(codes)
        labels: dict[str, str] = {}
        labels.update({code: _CODE_LABEL_CACHE[code] for code in normalized if code in _CODE_LABEL_CACHE})
        if self.ak is not None:
            for code in normalized:
                if code in labels:
                    continue
                name = self._load_single_code_label(code)
                if name and name not in {code, f"股票{code}"}:
                    labels[code] = str(name)
                    _CODE_LABEL_CACHE[code] = str(name)
        for code in normalized:
            labels.setdefault(code, code)
        return labels

    def _load_single_code_label(self, code: str) -> str | None:
        try:
            info = self.ak.stock_individual_info_em(symbol=code)
            values = dict(zip(info["item"].astype(str), info["value"]))
            name = values.get("股票简称")
            if name:
                return str(name)
        except Exception:
            pass
        profile = {"name": f"股票{code}", "industry": "未知", "data_warnings": []}
        self._enrich_profile_from_cninfo(code, profile)
        name = profile.get("name")
        return str(name) if name else None

    def _enrich_financials(self, code: str, profile: dict) -> None:
        try:
            raw = self.ak.stock_financial_analysis_indicator(symbol=code, start_year=str(date.today().year - 4))
            if raw.empty:
                return
            df = raw.copy()
            df["日期"] = pd.to_datetime(df["日期"], errors="coerce")
            df = df.dropna(subset=["日期"]).sort_values("日期")
            latest = df.iloc[-1]
            financial = {
                "report_date": latest["日期"].date().isoformat(),
                "roe": _num(latest.get("净资产收益率(%)")),
                "weighted_roe": _num(latest.get("加权净资产收益率(%)")),
                "revenue_growth": _num(latest.get("主营业务收入增长率(%)")),
                "profit_growth": _num(latest.get("净利润增长率(%)")),
                "debt_ratio": _num(latest.get("资产负债率(%)")),
                "gross_margin": _num(latest.get("销售毛利率(%)")),
                "net_margin": _num(latest.get("销售净利率(%)")),
                "cashflow_per_share": _num(latest.get("每股经营性现金流(元)")),
                "eps": _num(latest.get("摊薄每股收益(元)")),
            }
            profile["financial"] = financial
            profile["roe"] = financial["roe"] or financial["weighted_roe"]
            profile["debt_ratio"] = financial["debt_ratio"]
            profile["revenue_growth"] = financial["revenue_growth"]
            profile["profit_growth"] = financial["profit_growth"]
        except Exception as exc:
            profile["data_warnings"].append(f"财务指标接口不可用：{type(exc).__name__}")

    def _enrich_fund_flow(self, code: str, profile: dict) -> None:
        errors: list[str] = []
        loaders = [
            ("东方财富资金流直连", lambda: self._load_fund_flow_eastmoney(code)),
            ("AKShare资金流", lambda: self.ak.stock_individual_fund_flow(stock=code, market=market_prefix(code))),
        ]
        for source, loader in loaders:
            try:
                raw = loader()
                df = self._normalize_fund_flow(raw)
                if not df.empty:
                    profile["fund_flow"] = df.tail(80)
                    profile["fund_flow_source"] = source
                    return
            except Exception as exc:
                errors.append(f"{source}:{type(exc).__name__}")
        profile["data_warnings"].append(f"资金流数据源全部失败：{'；'.join(errors)}")

    def _load_fund_flow_eastmoney(self, code: str) -> pd.DataFrame:
        market_map = {"sh": 1, "sz": 0, "bj": 0}
        url = "https://push2his.eastmoney.com/api/qt/stock/fflow/daykline/get"
        params = {
            "lmt": "0",
            "klt": "101",
            "secid": f"{market_map[market_prefix(code)]}.{code}",
            "fields1": "f1,f2,f3,f7",
            "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61,f62,f63,f64,f65",
            "ut": "b2884a393a59ad64002292a3e90d46a5",
            "_": int(time.time() * 1000),
        }
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/122.0 Safari/537.36",
            "Referer": "https://data.eastmoney.com/zjlx/detail.html",
        }
        data = _request_json(url, params=params, headers=headers)
        klines = ((data or {}).get("data") or {}).get("klines") or []
        if not klines:
            return pd.DataFrame()
        df = pd.DataFrame([item.split(",") for item in klines])
        df.columns = [
            "日期",
            "主力净流入-净额",
            "小单净流入-净额",
            "中单净流入-净额",
            "大单净流入-净额",
            "超大单净流入-净额",
            "主力净流入-净占比",
            "小单净流入-净占比",
            "中单净流入-净占比",
            "大单净流入-净占比",
            "超大单净流入-净占比",
            "收盘价",
            "涨跌幅",
            "_1",
            "_2",
        ]
        return df

    def _normalize_fund_flow(self, raw: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(raw, pd.DataFrame) or raw.empty:
            return pd.DataFrame()
        df = raw.copy()
        keep = [
            "日期",
            "收盘价",
            "涨跌幅",
            "主力净流入-净额",
            "主力净流入-净占比",
            "超大单净流入-净额",
            "超大单净流入-净占比",
            "大单净流入-净额",
            "大单净流入-净占比",
            "中单净流入-净额",
            "中单净流入-净占比",
            "小单净流入-净额",
            "小单净流入-净占比",
        ]
        for col in keep:
            if col not in df:
                df[col] = pd.NaT if col == "日期" else None
        df = df[keep].copy()
        df["日期"] = pd.to_datetime(df["日期"], errors="coerce")
        for col in keep:
            if col != "日期":
                df[col] = pd.to_numeric(df[col], errors="coerce")
        return df.dropna(subset=["日期"]).sort_values("日期")

    def _enrich_news(self, code: str, profile: dict) -> None:
        errors: list[str] = []
        loaders = [
            ("东方财富新闻直连", lambda: self._load_news_eastmoney(code, profile)),
            ("AKShare东方财富新闻", lambda: self.ak.stock_news_em(symbol=code)),
            ("财新数据通", lambda: self.ak.stock_news_main_cx()),
        ]
        for source, loader in loaders:
            try:
                raw = loader()
                if isinstance(raw, pd.DataFrame) and not raw.empty:
                    filtered = self._filter_news_frame(raw, code, profile)
                    if not filtered.empty:
                        profile["news"] = filtered.head(12).copy()
                        profile["news_source"] = source
                        return
            except Exception as exc:
                errors.append(f"{source}:{type(exc).__name__}")

        profile["data_notes"] = profile.get("data_notes", [])
        detail = "、".join(dict.fromkeys(errors)) or "无返回"
        profile["data_warnings"].append(f"新闻数据源全部失败：{detail}")

    def _load_news_eastmoney(self, code: str, profile: dict) -> pd.DataFrame:
        callback = f"jQuery{int(time.time() * 1000)}"
        keyword = code
        name = str(profile.get("name") or "")
        if name and name not in {code, f"股票{code}"}:
            keyword = name
        inner_param = {
            "uid": "",
            "keyword": keyword,
            "type": ["cmsArticleWebOld"],
            "client": "web",
            "clientType": "web",
            "clientVersion": "curr",
            "param": {
                "cmsArticleWebOld": {
                    "searchScope": "default",
                    "sort": "default",
                    "pageIndex": 1,
                    "pageSize": 20,
                    "preTag": "",
                    "postTag": "",
                }
            },
        }
        url = "https://search-api-web.eastmoney.com/search/jsonp"
        params = {
            "cb": callback,
            "param": json.dumps(inner_param, ensure_ascii=False),
            "_": int(time.time() * 1000),
        }
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/122.0 Safari/537.36",
            "Referer": f"https://so.eastmoney.com/news/s?keyword={keyword}",
        }
        text = _request_text(url, params=params, headers=headers)
        left = text.find("(")
        right = text.rfind(")")
        if left == -1 or right == -1 or right <= left:
            raise ValueError("invalid eastmoney jsonp")
        data = json.loads(text[left + 1 : right])
        rows = ((data or {}).get("result") or {}).get("cmsArticleWebOld") or []
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows)
        link_code = df.get("code", pd.Series([""] * len(df))).astype(str)
        df["新闻链接"] = "http://finance.eastmoney.com/a/" + link_code + ".html"
        df = df.rename(
            columns={
                "date": "发布时间",
                "mediaName": "文章来源",
                "title": "新闻标题",
                "content": "新闻内容",
            }
        )
        for col in ["新闻标题", "新闻内容"]:
            if col in df:
                df[col] = (
                    df[col]
                    .astype(str)
                    .str.replace(r"<[^>]+>", "", regex=True)
                    .str.replace(r"\u3000", "", regex=True)
                    .str.replace(r"\r\n", " ", regex=True)
                )
        df["关键词"] = keyword
        keep = [col for col in ["关键词", "新闻标题", "新闻内容", "发布时间", "文章来源", "新闻链接"] if col in df.columns]
        return df[keep].copy() if keep else df

    def _filter_news_frame(self, raw: pd.DataFrame, code: str, profile: dict) -> pd.DataFrame:
        df = raw.copy()
        name = str(profile.get("name") or "")
        text_cols = [
            col
            for col in ["新闻标题", "标题", "内容", "新闻内容", "摘要", "文章来源", "source", "title"]
            if col in df.columns
        ]
        if not text_cols:
            text_cols = list(df.columns[: min(3, len(df.columns))])
        text = df[text_cols].astype(str).agg(" ".join, axis=1)
        mask = text.str.contains(code, regex=False, na=False)
        if name and name != code:
            mask = mask | text.str.contains(name, regex=False, na=False)
        filtered = df[mask].copy()
        return filtered if not filtered.empty else df

    def _enrich_notices(self, code: str, profile: dict) -> None:
        start = (date.today() - timedelta(days=120)).strftime("%Y-%m-%d")
        end = date.today().strftime("%Y-%m-%d")
        try:
            raw = self.ak.stock_individual_notice_report(
                security=code,
                symbol="全部",
                begin_date=start,
                end_date=end,
            )
            if not raw.empty:
                profile["notices"] = raw.head(12).copy()
                profile["notice_source"] = "AKShare个股公告"
                return
        except Exception:
            pass
        try:
            raw = self.ak.stock_zh_a_disclosure_report_cninfo(
                symbol=code,
                market="沪深京",
                start_date=start.replace("-", ""),
                end_date=end.replace("-", ""),
            )
            if not raw.empty:
                profile["notices"] = raw.head(12).copy()
                profile["notice_source"] = "巨潮公告"
        except Exception as exc:
            profile["data_warnings"].append(f"公告接口不可用：{type(exc).__name__}")

    def load_industry_boards(self) -> tuple[pd.DataFrame, str]:
        if self.ak is None:
            return pd.DataFrame(), "未安装 AKShare"
        for source, loader in [
            ("同花顺行业列表", self.ak.stock_board_industry_name_ths),
            ("东方财富行业列表", self.ak.stock_board_industry_name_em),
        ]:
            try:
                df = loader()
                if not df.empty:
                    return df, source
            except Exception:
                continue
        return pd.DataFrame(), "行业接口不可用"
