@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo Запуск скачивания модели...
python download_model.py
if errorlevel 1 pause
pause
