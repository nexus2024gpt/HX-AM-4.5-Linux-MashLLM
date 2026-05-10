@echo off
chcp 65001 >nul
title HX-AM v4.5

echo ========================================
echo    Starting HX-AM v4.5...
echo ========================================

wsl -d Ubuntu -u roman220877 --cd /home/roman220877/hxam bash -c "source venv/bin/activate && python hxam_v_4_server.py"

echo.
echo ========================================
echo Server has stopped.
pause