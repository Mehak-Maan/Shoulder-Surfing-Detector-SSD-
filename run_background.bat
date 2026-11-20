@echo off
cd /d "%~dp0"
:: Starts in background without any black console window
start "" ".venv\Scripts\pythonw.exe" main.py --mode headless
