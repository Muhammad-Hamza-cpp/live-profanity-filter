@echo off
rem Emergency fix: if sound is silent because the filter crashed, this puts your real speakers back as default.
powershell -NoProfile -ExecutionPolicy Bypass -Command "$s='%~dp0audio_default.ps1'; $d = & $s list | Where-Object { $_ -notmatch 'VB-Audio' } | Select-Object -First 1; & $s set ($d -split '\|')[0]; Write-Host ('Default output set to ' + ($d -split '\|')[1])"
pause
