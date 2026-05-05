@echo off
title Stop HX-AM v4.5
echo Stopping HX-AM v4.5 in WSL2...
wsl bash -c "pkill -f 'hxam_v_4_server.py'"
echo Done.
timeout /t 2 >nul
exit