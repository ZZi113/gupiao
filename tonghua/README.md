# A股个人投资操作助手

一个可本地运行、也可部署成网站的 A 股投资辅助工具，用来帮助你做自选股扫描、个股操作分析、持仓风险检查和每日操作清单。

> 重要提示：本项目只用于个人研究和辅助决策，不构成投资建议。任何买卖决定都需要你结合自身风险承受能力独立判断。

## 主要功能

- 实时行情：新浪日线 + 分钟线，尽量合成当天最新 K 线
- 自动刷新：可选 30/60/120/300 秒刷新一次行情，适合盯盘时使用
- 个股操作建议：可买入、小仓试探、观察等待、继续持有、建议减仓、卖出/回避
- 资金流：个股主力资金近 1/5/20 日变化
- 财务指标：ROE、营收增长、净利润增长、资产负债率、现金流
- 新闻公告：展示近期新闻和公告，并识别常见风险词
- 简单回测：验证均线趋势规则过去是否有效
- 板块轮动：展示可用行业列表，并统计自选股行业分布
- 多智能体研判：借鉴 TradingAgents 的角色拆分思路，分别给出技术面、资金面、基本面、新闻公告和风险经理观点，再汇总成五档评级
- 优质股票发现：从东方财富/AKShare/新浪全市场快照中按流动性、估值、趋势、资金和风险做粗筛，再对前排候选做详细复核
- 大模型报告：支持 OpenAI-compatible 接口，可选配置

## 运行方式

```powershell
pip install -r requirements.txt
powershell -ExecutionPolicy Bypass -File .\run_app.ps1
```

或直接运行：

```powershell
python -m streamlit run app.py
```

打开浏览器访问：

```text
http://localhost:8501
```

## 部署成网站

本项目已经整理成 Streamlit Cloud 可部署结构。把项目上传到 GitHub 后，在 Streamlit Cloud 里选择：

```text
入口文件：app.py
Python：3.12
依赖文件：requirements.txt
```

详细步骤见 [DEPLOY.md](DEPLOY.md)。

如果要给网站加访问密码，在 Streamlit Cloud 的 Secrets 中设置：

```toml
APP_PASSWORD = "你自己的密码"
```

如果要设置默认股票池，也可以在 Secrets 中加入：

```toml
APP_DEFAULT_CODES = "603906,600410"
```

页面左侧也有 `保存股票池` 按钮。保存后，把当前网址收藏起来，下次打开会自动带上这些股票。

## 推荐使用流程

1. 在“自选股扫描”输入你关注的股票代码，比如 `600519, 000001, 300750`。
2. 查看操作标签和关键原因。
3. 对想进一步看的股票进入“个股分析”。
4. 到“多智能体研判”查看不同角色的分歧、五档评级和交易员执行单。
5. 到“优质股票发现”扫描新的候选池，把看中的候选加入自选股继续跟踪。
6. 盯盘时打开侧边栏的“自动刷新行情”，选择 30 秒或 60 秒刷新间隔。
7. 如果已经持有，在侧边栏填写成本价和仓位，再看持仓建议。

## 优质股票发现是怎么筛的

这个功能借鉴 Qlib、FinRL 和 TradingAgents 这类开源项目常见的“先全市场打分，再对少数候选复核”的流程：

1. 先取东方财富全A实时快照，失败时再用 AKShare 东方财富全市场快照和 AKShare 新浪全A快照，全都失败才退到演示候选池。
2. 先排除 ST、退市、新股标记、成交过低、极端估值、涨跌停附近和换手异常的股票。
3. 初筛分综合 PE/PB、成交额、市值、60日涨跌幅、当日涨跌幅、换手率、量比、振幅、主力净流入等因子。
4. 可切换 7 种筛选风格：稳健优质、趋势增强、低估修复、成长质量、资金关注、突破跟踪、超跌修复。
5. 对初筛前排再跑个股分析和多角色复核，最后给出“候选研究名单”，不是直接买卖指令。

## 大模型报告

侧边栏可以填写 OpenAI-compatible 接口：

- 接口地址：例如 `https://api.openai.com/v1`，或其他兼容服务商地址
- 模型名：例如 `gpt-4o-mini`
- API Key：只在当前页面会话中使用，不写入文件

如果不填，系统会使用规则版报告。

## 每日报告与推送

手动生成报告：

```powershell
python scripts\daily_report.py --codes "600519,000001,300750"
```

报告会生成在 `reports` 目录。

如果要企业微信机器人推送：

```powershell
$env:WECHAT_WEBHOOK_URL="你的机器人Webhook"
python scripts\daily_report.py --codes "600519,000001,300750" --push
```

如果要邮件推送，需要设置：

```powershell
$env:SMTP_HOST="smtp.example.com"
$env:SMTP_PORT="465"
$env:SMTP_USER="your@email.com"
$env:SMTP_PASSWORD="邮箱授权码"
$env:REPORT_EMAIL_TO="to@email.com"
python scripts\daily_report.py --codes "600519,000001,300750" --push
```

可以用 Windows 任务计划程序每天收盘后运行上面的命令。
