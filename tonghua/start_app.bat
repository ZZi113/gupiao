@echo off
chcp 65001 >nul
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1
echo ========================================
echo A股个人投资操作助手 - 本地启动
echo ========================================
echo 项目目录: %cd%
echo.
echo 正在检查 Python...
python --version
if errorlevel 1 (
    echo.
    echo 没有找到 python，请先打开 Anaconda Prompt 后再运行本文件。
    pause
    exit /b 1
)
echo.
echo 正在检查 Streamlit...
python -m streamlit --version
if errorlevel 1 (
    echo.
    echo 没有找到 streamlit。请先安装依赖：
    echo pip install -r requirements.txt
    pause
    exit /b 1
)
echo.
echo 正在启动网站，请保持这个黑色窗口不要关闭。
echo 浏览器地址: http://localhost:8501
echo.
python -m streamlit run app.py --server.address localhost --server.port 8501 --browser.gatherUsageStats false
echo.
echo Streamlit 已停止。上面如果有红色报错，请截图发给我。
echo 按任意键关闭窗口。
pause >nul
