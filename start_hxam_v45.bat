@echo off
chcp 65001 >nul
title HX-AM v4.5.2

echo ========================================
echo    HX-AM v4.5.2 — MashAdapter Edition
echo ========================================

:: ── Шаг 1: Проверить что mesh-llm запущен ──────────────────────────
echo.
echo [1/3] Проверка MashLLM...
wsl curl -s --max-time 3 http://localhost:9337/v1/models >nul 2>&1
if %errorlevel% equ 0 (
    echo [OK] MashLLM OpenAI API отвечает на :9337
) else (
    echo [WARN] MashLLM на :9337 недоступен.
    echo        HX-AM запустится, но будет использовать только резервные провайдеры.
    echo        Чтобы запустить MashLLM: start_mesh.bat
)

:: ── Шаг 2: Проверить Management API ────────────────────────────────
wsl curl -s --max-time 3 http://localhost:3131/api/status >nul 2>&1
if %errorlevel% equ 0 (
    echo [OK] MashLLM Management API отвечает на :3131
) else (
    echo [INFO] Management API на :3131 недоступен — адаптер будет использовать /v1/models
)

:: ── Шаг 3: Запустить HX-AM ─────────────────────────────────────────
echo.
echo [2/3] Запуск HX-AM v4.5.2 в WSL2...
echo       Остановка: Ctrl+C или stop_hxam_v45.bat
echo.

wsl -d Ubuntu -u roman220877 --cd /home/roman220877/hxam ^
    bash -c "source venv/bin/activate && python hxam_v_4_server.py 2>&1 | tee /tmp/hxam_last.log"

:: ── Завершение ──────────────────────────────────────────────────────
echo.
echo ========================================
echo [3/3] Сервер остановлен.
echo       Лог: wsl cat /tmp/hxam_last.log
pause