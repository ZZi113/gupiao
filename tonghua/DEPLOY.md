# 部署到 Streamlit Cloud

目标：把本地项目变成一个别人可以直接打开的网址。

## 1. 上传到 GitHub

1. 登录 GitHub。
2. 新建仓库，例如 `ashare-advisor`。
3. 把本项目目录里的文件上传到仓库。

需要包含这些关键文件：

```text
app.py
requirements.txt
README.md
src/
scripts/
.streamlit/config.toml
.streamlit/secrets.toml.example
.gitignore
```

不要上传这些目录：

```text
logs/
reports/
__pycache__/
.streamlit/secrets.toml
```

## 2. 创建 Streamlit Cloud 应用

1. 打开 `https://share.streamlit.io`。
2. 用 GitHub 登录。
3. 点击 `Create app`。
4. 选择你的 GitHub 仓库。
5. Branch 选择 `main`。
6. Main file path 填：

```text
app.py
```

7. Python 版本建议选择 `3.12`。
8. 点击 Deploy。

部署完成后，你会得到类似这样的地址：

```text
https://your-name-ashare-advisor.streamlit.app
```

别人打开这个地址就可以使用。

## 3. 可选：设置访问密码

如果你不想公开给所有人使用，在 Streamlit Cloud 的应用设置里找到 `Secrets`，填入：

```toml
APP_PASSWORD = "你自己的密码"
```

设置后，用户打开网站需要先输入密码。

## 4. 可选：设置大模型报告

如果你要让云端直接生成大模型报告，也在 `Secrets` 里加入：

```toml
LLM_BASE_URL = "https://api.openai.com/v1"
LLM_MODEL = "gpt-4o-mini"
LLM_API_KEY = "你的API Key"
```

如果不设置，系统会使用规则版报告。

## 5. 注意事项

- 本项目仅供研究和复盘，不构成投资建议。
- 免费托管平台可能会休眠，第一次打开可能比较慢。
- AKShare 的部分数据接口可能受网络和源站限制影响，页面中会显示数据源和接口警告。
- 如果你的仓库是公开的，不要把真实 API Key 或密码写进代码。

