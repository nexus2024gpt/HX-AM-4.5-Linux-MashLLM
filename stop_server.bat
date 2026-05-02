@echo off
title Stop HX-AM v4.2
cd /d "D:\Projects\HX-AM-Proxy-v4.2-Dual-LLM-4Dgraf-MathCore"
echo Searching for python process on port 8000...
for /f "tokens=5" %%a in ('netstat -aon ^| find ":8000" ^| find "LISTENING"') do set PID=%%a
if defined PID (
    echo Stopping process with PID: %PID%
    taskkill /PID %PID% /F
    echo Server stopped.
) else (
    echo No server process found on port 8000.
)
timeout /t 2 >nul
exit