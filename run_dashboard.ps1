$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"

if (-not (Test-Path $Python)) {
    throw "Could not find .venv Python at $Python"
}

Set-Location $ProjectRoot
& $Python -m streamlit run dashboard\app.py --server.address 127.0.0.1 --server.port 8501
