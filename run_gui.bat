@echo off
title Shoulder Surfing Detector (GUI Mode)
cd /d "%~dp0"
echo Starting Shoulder Surfing Detector in GUI mode...
.venv\Scripts\python.exe main.py --mode gui
pause
