@echo off
echo Syncing Linux (WSL2) → Windows...
wsl rsync -avh --progress --exclude 'venv/' /home/roman220877/hxam/ /mnt/d/Projects/HX-AM-Proxy-v4.2-Dual-LLM-4Dgraf-MathCore/
echo Done.
timeout /t 2 >nul
exit /b
