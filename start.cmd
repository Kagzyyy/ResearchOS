@echo off
REM Start ResearchOS: backend on :8100, web app on :3000.
REM The backend loads 40k passages and two models, so the first response takes
REM a couple of minutes. The web app shows "Backend not reachable" until then.
REM That is expected on a cold start, not an error.

cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
  echo Virtual environment missing. Run:  uv venv --python 3.11
  exit /b 1
)
if not exist "data\arxiv_chunks.jsonl" (
  echo No corpus found. Run:  python scripts\ingest_arxiv.py --config configs\ml.yaml
  exit /b 1
)
if not exist "web\node_modules" (
  echo Web dependencies missing. Run:  cd web ^&^& npm install
  exit /b 1
)

echo Starting backend  http://127.0.0.1:8100
start "ResearchOS backend" .venv\Scripts\python.exe scripts\serve_ml.py --config configs\ml.yaml

echo Starting web app  http://localhost:3000
start "ResearchOS web" cmd /c "cd web && npm run dev"

echo.
echo   Web app   http://localhost:3000
echo   Health    http://127.0.0.1:8100/health
echo.
echo The backend needs 2-3 minutes to load its models on a cold start.
echo Check it is ready with:  python scripts\healthcheck.py
