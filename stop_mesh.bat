@echo off
title Stop mesh-llm
echo Stopping mesh-llm in WSL2...
wsl bash -c "pkill -f 'mesh-llm serve'"
echo Done.
timeout /t 2 >nul
exit