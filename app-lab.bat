@echo off
rem Starts the desktop app from the lab environment (.venv-lab), which adds the experimental
rem "Ultra-low lag - trained detector" mode. See docs/mute-detector.md.
cd /d "%~dp0"
start "" ".venv-lab\Scripts\pythonw.exe" app.pyw
