@echo off
title Start HX-AM v4.5 Server
echo [INFO] Opening Ubuntu terminal with HX-AM v4.5...
:: Запускаем Ubuntu, выполняем команды и оставляем окно открытым
start "HX-AM v4.5" ubuntu.exe run bash -c "cd ~/hxam && source venv/bin/activate && python hxam_v_4_server.py; echo 'Server stopped. Press Enter...'; read"
echo [OK] Command sent. The server window should appear shortly.
timeout /t 3 >nul
exit