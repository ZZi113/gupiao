$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $PSScriptRoot
New-Item -ItemType Directory -Force logs | Out-Null
"Starting at $(Get-Date)" | Out-File -FilePath logs\run_app.log -Encoding utf8
python --version *>> logs\run_app.log
python -m streamlit run app.py --server.headless true --server.address localhost --server.port 8501 --browser.gatherUsageStats false *>> logs\run_app.log
