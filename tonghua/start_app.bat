@echo off
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1

echo ========================================
echo A-share advisor local launcher
echo ========================================
echo Project folder:
echo %cd%
echo.

echo Checking Python...
python --version
if errorlevel 1 (
    echo.
    echo ERROR: Python was not found.
    echo Please install Python or open this project from Anaconda Prompt.
    echo.
    pause
    exit /b 1
)

echo.
echo Checking Streamlit...
python -m streamlit --version
if errorlevel 1 (
    echo.
    echo ERROR: Streamlit was not found.
    echo Run this command first:
    echo pip install -r requirements.txt
    echo.
    pause
    exit /b 1
)

echo.
echo Starting Streamlit...
echo Keep this window open while using the app.
echo Open this URL in your browser:
echo http://localhost:8501
echo.

python -m streamlit run app.py --server.address localhost --server.port 8501 --browser.gatherUsageStats false

echo.
echo Streamlit stopped. If there is an error above, take a screenshot and send it to Codex.
echo.
pause
