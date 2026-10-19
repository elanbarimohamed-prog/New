@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo First create the environment and install requirements. See README.md.
  pause
  exit /b 1
)
".venv\Scripts\python.exe" -m streamlit run streamlit_app.py
