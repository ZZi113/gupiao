from __future__ import annotations

import argparse
import os
import smtplib
import sys
from datetime import date
from email.message import EmailMessage
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ashare_advisor.data import DataProvider, normalize_codes
from src.ashare_advisor.reporting import build_rule_report
from src.ashare_advisor.rules import analyze_stock


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate an A-share daily operation report.")
    parser.add_argument("--codes", required=True, help="Comma separated stock codes, e.g. 600519,000001")
    parser.add_argument("--days", type=int, default=260)
    parser.add_argument("--output", default="")
    parser.add_argument("--push", action="store_true", help="Push through env-configured webhook/email.")
    args = parser.parse_args()

    codes = normalize_codes(args.codes)
    provider = DataProvider()
    reports = [f"# A股每日操作清单 - {date.today().isoformat()}", ""]
    for code in codes:
        df, profile, source = provider.load_stock(code, days=args.days, realtime=True)
        result = analyze_stock(code, df, profile)
        result["source"] = source
        reports.append(build_rule_report(result))
        reports.append("\n---\n")

    content = "\n".join(reports)
    output = Path(args.output) if args.output else Path("reports") / f"daily_{date.today().strftime('%Y%m%d')}.md"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(content, encoding="utf-8")
    print(output)

    if args.push:
        push_webhook(content)
        push_email(content)


def push_webhook(content: str) -> None:
    url = os.getenv("WECHAT_WEBHOOK_URL") or os.getenv("REPORT_WEBHOOK_URL")
    if not url:
        return
    payload = {"msgtype": "markdown", "markdown": {"content": content[:3900]}}
    requests.post(url, json=payload, timeout=20).raise_for_status()


def push_email(content: str) -> None:
    host = os.getenv("SMTP_HOST")
    to_addr = os.getenv("REPORT_EMAIL_TO")
    user = os.getenv("SMTP_USER")
    password = os.getenv("SMTP_PASSWORD")
    if not host or not to_addr or not user or not password:
        return
    port = int(os.getenv("SMTP_PORT", "465"))
    msg = EmailMessage()
    msg["Subject"] = f"A股每日操作清单 {date.today().isoformat()}"
    msg["From"] = user
    msg["To"] = to_addr
    msg.set_content(content)
    with smtplib.SMTP_SSL(host, port) as smtp:
        smtp.login(user, password)
        smtp.send_message(msg)


if __name__ == "__main__":
    main()
