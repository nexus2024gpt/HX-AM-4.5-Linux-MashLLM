# mgap_lib/scripts/server_integration_patch.py
"""
Патч для интеграции MGAP Library в hxam_v_4_server.py.

ВАЖНО: Это инструкция, а не автоматический патч.
Скопируй три блока в hxam_v_4_server.py вручную.

============================================================
БЛОК 1 — Добавить в импорты (после строки с MGAPMatcher):
============================================================

# MGAP Library v1.0
from mgap_lib.api.routes import mgap_router
from mgap_lib.api.dependencies import init_engine as mgap_init_engine

============================================================
БЛОК 2 — Добавить после строки init_engine (около строки 70):
============================================================

# Инициализация MGAP Library (замена прямого MGAPMatcher)
mgap_init_engine(
    registry_path="mgap_registry.json",
    artifacts_dir="artifacts",
    use_llm=True,
    gap_mode="max",
)
app.include_router(mgap_router)
logger.info("MGAP Library v1.0 router подключён на /mgap/...")

============================================================
БЛОК 3 — Удалить или закомментировать старые эндпоинты:
============================================================
# Старые эндпоинты /mgap/match, /mgap/registry, /mgap/batch
# теперь заменены роутером mgap_lib.
# Если нужна обратная совместимость — оставь старые,
# они не конфликтуют (другой путь).

============================================================
ПРОВЕРКА ИНТЕГРАЦИИ:
============================================================
После перезапуска сервера:

  python -c "import requests; r=requests.get('http://localhost:8000/mgap/stats'); print(r.json())"

Ожидаемый ответ:
  {"registry": {"total_models": 11, ...}, "db_available": false, ...}

============================================================
ТЕСТ НОВОГО CLI (из папки проекта):
============================================================
  python mgap_lib/cli/mgap_cli.py registry
  python mgap_lib/cli/mgap_cli.py classify --domain biology
  python mgap_lib/cli/mgap_cli.py classify --domains "biology,economics,physics,neuroscience"
  python mgap_lib/cli/mgap_cli.py stats
  python mgap_lib/cli/mgap_cli.py match --artifact 32d4aa917ac4

============================================================
ТЕСТ ИНИЦИАЛИЗАЦИИ БД (опционально):
============================================================
  python mgap_lib/scripts/init_db.py
  # или
  python mgap_lib/cli/mgap_cli.py init-db

После инициализации БД при запуске сервера передавать db_url:
  mgap_init_engine(
      registry_path="mgap_registry.json",
      db_url="sqlite:///mgap.db",
      artifacts_dir="artifacts",
  )
"""

print(__doc__)
