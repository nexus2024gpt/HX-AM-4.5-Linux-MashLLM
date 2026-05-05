@echo off
title mesh-llm auto
echo Checking if mesh-llm is already running...
wsl curl -s --max-time 2 http://localhost:9337/v1/models >nul 2>&1
if %errorlevel% equ 0 (
    echo [OK] mesh-llm is already running and responding on port 9337.
    echo Opening a window with live logs...
    start "mesh-llm logs" wsl bash -c "tail -f ~/mesh-llm.log"
    timeout /t 3 >nul
    exit
)
echo mesh-llm not running. Launching in a new window...
start "mesh-llm" wsl bash -c "/home/roman220877/.local/bin/mesh-llm serve --auto --port 9337; exec bash"
echo mesh-llm window opened. Wait until you see "Uvicorn running".
pause
exit