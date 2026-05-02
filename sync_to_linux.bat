@echo off
echo Syncing Windows → Linux (WSL2)...
wsl rsync -avh --progress --exclude 'venv/' /mnt/d/Projects/HX-AM-Proxy-v4.2-Dual-LLM-4Dgraf-MathCore/ /home/roman220877/hxam/
echo Done.
timeout /t 2 >nul
exit /b
