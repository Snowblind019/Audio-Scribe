@echo off
rem Audio Scribe installer for Windows. Double-click this file.
rem Extra options are passed through, for example: install.bat -WithStems
setlocal
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0install.ps1" %*
echo.
pause
