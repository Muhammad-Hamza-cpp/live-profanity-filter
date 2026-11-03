@echo off
rem Starts the profanity filter. Press Ctrl+C (or close this window) to stop and get normal sound back.
rem Extra options can be added, e.g.:  start.bat --delay 2.5   or   start.bat --output Headphones
cd /d "%~dp0"
title Audio Censor
".venv\Scripts\python.exe" censor.py %*
pause
