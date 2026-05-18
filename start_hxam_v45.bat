@echo off
chcp 65001 >nul
title HX-AM v4.5.2 — Full Stack

echo ========================================
echo    HX-AM v4.5.2 — Полный запуск
echo ========================================

echo [1/4] Остановка старых процессов...
wsl -d Ubuntu -u roman220877 --cd /home/roman220877/hxam bash -c "pkill -f llama-server; pkill -f mesh-llm; pkill -f hxam_v_4_server.py" 2>nul

echo [2/4] Запуск llama-server...
wsl -d Ubuntu -u roman220877 --cd /home/roman220877/hxam ./start_llama_cpu.sh

echo [3/4] Запуск MashLLM...
wsl -d Ubuntu -u roman220877 --cd /home/roman220877/hxam bash -c "mesh-llm client --auto > mesh_llm.log 2>&1 &" 2>nul

echo [4/4] Запуск HX-AM...
wsl -d Ubuntu -u roman220877 --cd /home/roman220877/hxam ^
    bash -c "source venv/bin/activate && python hxam_v_4_server.py 2>&1 | tee /tmp/hxam_last.log"

echo.
echo ========================================
echo Все сервисы запущены!
echo HX-AM → http://127.0.0.1:8000
echo ========================================
echo.
pause