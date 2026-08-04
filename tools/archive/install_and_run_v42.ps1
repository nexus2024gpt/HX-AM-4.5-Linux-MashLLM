# ══════════════════════════════════════════════════════════════
# HX-AM v4.2 — установка зависимостей и запуск
# Запускать из папки проекта: cd путь\к\проекту
# ══════════════════════════════════════════════════════════════

# ШАГ 1 — Проверка Python
Write-Host "`n[1/5] Проверка Python..." -ForegroundColor Cyan
python --version
if ($LASTEXITCODE -ne 0) {
    Write-Host "Python не найден. Установи Python 3.10+ с python.org" -ForegroundColor Red
    exit 1
}

# ШАГ 2 — Виртуальное окружение
Write-Host "`n[2/5] Виртуальное окружение..." -ForegroundColor Cyan
if (-Not (Test-Path "venv")) {
    python -m venv venv
    Write-Host "Создано: venv\" -ForegroundColor Green
} else {
    Write-Host "Уже существует: venv\" -ForegroundColor Yellow
}

# ШАГ 3 — Зависимости
Write-Host "`n[3/5] Установка зависимостей v4.2..." -ForegroundColor Cyan
.\venv\Scripts\Activate.ps1
pip install --upgrade pip --quiet
pip install -r requirements_v42.txt

if ($LASTEXITCODE -ne 0) {
    Write-Host "Ошибка установки. Проверь requirements_v42.txt" -ForegroundColor Red
    exit 1
}
Write-Host "Все зависимости установлены." -ForegroundColor Green

# ШАГ 4 — Создание папок v4.2
Write-Host "`n[4/5] Подготовка папок v4.2..." -ForegroundColor Cyan
$folders = @("artifacts", "chat_history", "prompts", "schemas", "sim_results", "insights", "tools", "trash", "config")
foreach ($folder in $folders) {
    if (-Not (Test-Path $folder)) {
        New-Item -ItemType Directory -Path $folder | Out-Null
        Write-Host "Создана папка: $folder\" -ForegroundColor Green
    } else {
        Write-Host "Уже существует: $folder\" -ForegroundColor Yellow
    }
}

# ШАГ 5 — Тест MathCore
Write-Host "`n[5/5] Проверка MathCore..." -ForegroundColor Cyan
python math_core.py --test
if ($LASTEXITCODE -ne 0) {
    Write-Host "MathCore тест упал. Проверь зависимости scipy/numpy." -ForegroundColor Yellow
} else {
    Write-Host "MathCore OK" -ForegroundColor Green
}

Write-Host "`n✅ v4.2 готов! Запускаю сервер..." -ForegroundColor Green
Write-Host "   Открой браузер: http://localhost:8000" -ForegroundColor White
Write-Host "   API: /math/stress/{id}  /insights/feed  /math/stats" -ForegroundColor White
Write-Host "   Для остановки: Ctrl+C`n" -ForegroundColor White

python hxam_v_4_server.py
