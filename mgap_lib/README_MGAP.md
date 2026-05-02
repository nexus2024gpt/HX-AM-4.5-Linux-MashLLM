# MGAP Library v1.0
## Metric GAP — перенос инвариантов HX-AM в отраслевые системы

MGAP Library расширяет HX-AM v4.4: находит отраслевые модели (WMS, EEGlab, OpenFOAM и др.),
резонирующие с найденными инвариантами, вычисляет метрический разрыв и генерирует
готовый код мониторинга под конкретную программу.

---

## Структура

```
mgap_lib/
├── __init__.py
├── data/
│   ├── domain_map.json          # HX-AM домены → UNESCO (22 exact + 35 keyword)
│   └── unesco_taxonomy.json     # UNESCO 1988: 6 дисциплин, 28 секторов, 120+ специализаций
├── config/
│   └── settings.py              # Настройки из .env
├── engine/
│   ├── domain_classifier.py     # Tier1/2/3 классификация без LLM
│   ├── gap_calculator.py        # η/τ/K Gap + уровни риска
│   ├── registry.py              # Загрузка из DB или JSON-фолбэк
│   └── matcher.py               # MGAPEngine — оркестратор
├── models/
│   └── database.py              # SQLAlchemy ORM (Discipline→Sector→Specialization→Model→Run)
├── api/
│   ├── dependencies.py          # FastAPI DI: get_engine, get_session
│   └── routes.py                # 9 эндпоинтов /mgap/...
├── cli/
│   └── mgap_cli.py              # CLI: match|batch|registry|classify|stats|init-db
├── scripts/
│   ├── init_db.py               # Инициализация БД
│   └── server_integration_patch.py  # Инструкция интеграции в сервер
└── requirements_mgap.txt
```

---

## Быстрый старт

### 1. Установка

```powershell
# Из папки проекта
pip install sqlalchemy --break-system-packages
```

### 2. Тест без БД (JSON-фолбэк)

```powershell
python mgap_lib/cli/mgap_cli.py stats
python mgap_lib/cli/mgap_cli.py classify --domains "biology,economics,geology,neuroscience"
python mgap_lib/cli/mgap_cli.py match --artifact 32d4aa917ac4
```

### 3. Инициализация БД (опционально)

```powershell
python mgap_lib/scripts/init_db.py
# или через CLI:
python mgap_lib/cli/mgap_cli.py init-db
```

### 4. Из Python

```python
from mgap_lib.engine.matcher import MGAPEngine

engine  = MGAPEngine.from_json("mgap_registry.json")
results = engine.match_artifact("32d4aa917ac4", top_k=3)

for r in results:
    print(r["model_id"], r["resonance"], r["gap"]["risk_level"])
    print(r["domain_classification"]["sector_name_ru"])
```

---

## Интеграция в hxam_v_4_server.py

Добавить три строки в `hxam_v_4_server.py` (после импортов MGAPMatcher):

```python
# MGAP Library v1.0
from mgap_lib.api.routes import mgap_router
from mgap_lib.api.dependencies import init_engine as mgap_init_engine

# После создания FastAPI app:
mgap_init_engine(registry_path="mgap_registry.json", artifacts_dir="artifacts")
app.include_router(mgap_router)
```

После этого доступны новые эндпоинты:

| Метод | Путь | Описание |
|-------|------|----------|
| `GET` | `/mgap/match/{id}` | Топ-K моделей для артефакта |
| `POST` | `/mgap/match` | Матч с inline JSON |
| `POST` | `/mgap/batch` | Все артефакты |
| `GET` | `/mgap/registry` | Список моделей |
| `GET` | `/mgap/model/{id}` | Полная модель |
| `GET` | `/mgap/classify?domain=biology` | UNESCO-классификация |
| `GET` | `/mgap/taxonomy` | Полное дерево UNESCO |
| `GET` | `/mgap/runs` | История из БД |
| `GET` | `/mgap/stats` | Статистика |

---

## Классификатор доменов (без LLM)

Три уровня без сетевых запросов:

```
Tier 1 (exact):   "biology"    → (1, "1.5", "Молекулярная биология")  — 100% conf
Tier 2 (keyword): "neuro..."   → (3, "3.1", "Нейрофизиология")        — 80% conf
Tier 3 (cosine):  any string   → ближайший сектор UNESCO via MiniLM    — ~0.3–0.9
```

```python
from mgap_lib.engine.domain_classifier import DomainClassifier
clf = DomainClassifier()
print(clf.describe("ecology"))
# [4/4.2] Сельскохозяйственные науки → Экология → Популяционная динамика (method=exact, conf=1.0)
```

---

## Gap Calculator

```
eta_gap  = max(0, (artifact_eta - model_eta_max) / model_eta_max)
tau_gap  = max(0, (artifact_tau - model_tau_max) / model_tau_max)
K_gap    = max(0, (model_K_min  - artifact_K)   / model_K_min)

composite (mode=max)  = max(eta_gap, tau_gap, K_gap)

Risk levels:
  none:     composite = 0
  monitor:  0 < composite ≤ 0.20
  moderate: 0.20 < composite ≤ 0.50
  critical: composite > 0.50
```

---

## Реестр моделей (mgap_registry.json)

11 моделей в 6 предметных областях:

| ID | Название | Логия | math_type | Программы |
|----|----------|-------|-----------|-----------|
| M1 | Kanban (pull) | Логистика | graph_invariant | WMS, SAP EWM |
| M2 | Маршрутизация с задержками | Логистика | kuramoto | OptiFlow |
| M3 | Фазовая синхронизация | Нейронауки | kuramoto | EEGlab |
| M4 | Спайковая сеть (SNN) | Нейронауки | kuramoto | SpiNNaker |
| M5 | Межбанковские кредиты | Финансы | graph_invariant | Bloomberg |
| M6 | Самуэльсон-Хикс | Финансы | delay | EViews |
| M7 | Лотки-Вольтерра | Экология | delay | Populus |
| M8 | Поток энергии | Экология | graph_invariant | Ecopath |
| M9 | Распространение мемов | Социология | kuramoto | Gephi |
| M10 | Гранулярный хаос | Социология | graph_invariant | NetLogo |
| M11 | Navier-Stokes | Гидродинамика | delay | OpenFOAM |

### Добавление новой модели

Добавить объект в `mgap_registry.json` → `models[]`:

```json
{
  "id": "M12",
  "logia": "Медицина",
  "industry": "Фармакокинетика",
  "name": "Двухкамерная PK-модель",
  "math_type": "delay",
  "disc_code": "3",
  "sector_code": "3.2",
  "programs": ["NONMEM", "Monolix", "Phoenix WinNonlin"],
  "four_d_matrix": {...},
  "expected_ranges": {"tau": [0.5, 8.0], "K": [0.1, 0.6], "eta": [0.05, 0.4]},
  "weights": {"tau": 0.5, "K": 0.3, "eta": 0.2},
  "critical_thresholds": {"eta_max": 0.35, "tau_max": 6.0, "K_min": 0.1},
  "translation_map": {
    "tau": {"industry_term": "Период полувыведения (ч)", "description": "...", "typical_values": "1–24 ч"},
    "K":   {"industry_term": "Константа элиминации", "description": "...", "typical_values": "0.1–0.6"},
    "eta": {"industry_term": "Межиндивидуальная вариабельность CV", "description": "...", "typical_values": "0.1–0.5"}
  },
  "blind_spot_template": "Стандартные PK-модели не отслеживают критическую вариабельность η_max={eta_max}...",
  "math_adaptation_formula": "stability_margin = min(1 - eta/eta_crit, 1 - tau/tau_crit)",
  "example_data": {"type": "delay", "coupling_K": 0.3, "noise_eta": 0.2, "delay_tau": 2.5}
}
```

После добавления — пересинхронизировать БД:
```powershell
python mgap_lib/scripts/init_db.py
```

---

## Переменные окружения

```env
MGAP_DB_URL=sqlite:///mgap.db          # строка подключения
MGAP_REGISTRY_PATH=mgap_registry.json  # путь к реестру
MGAP_ARTIFACTS_DIR=artifacts           # папка артефактов
MGAP_USE_LLM=true                      # улучшать blind_spot через LLM
MGAP_DEFAULT_TOP_K=3                   # топ-K матчей
MGAP_MIN_RESONANCE=0.3                 # минимальный резонанс для batch
MGAP_GAP_MODE=max                      # max | mean | rms
```

---

## Производительность (i5-6300U / 8 GB)

| Операция | Время |
|----------|-------|
| Классификация Tier 1/2 | <1 мс |
| Классификация Tier 3 (cosine, первый раз) | ~200 мс (загрузка модели) |
| Классификация Tier 3 (cosine, кэш) | <5 мс |
| Матч 1 артефакта × 11 моделей | ~20 мс (без LLM) |
| Матч 1 артефакта × 11 моделей | ~2–5 сек (с LLM blind_spot) |
| Batch 50 артефактов | ~15 сек (без LLM) |

---

## Статус разработки

| Компонент | Статус |
|-----------|--------|
| `domain_classifier.py` | ✅ Готов (Tier 1/2/3) |
| `gap_calculator.py` | ✅ Готов |
| `registry.py` | ✅ Готов (JSON + DB) |
| `matcher.py` (MGAPEngine) | ✅ Готов |
| `database.py` (ORM) | ✅ Готов |
| `api/routes.py` | ✅ Готов (9 эндпоинтов) |
| `cli/mgap_cli.py` | ✅ Готов |
| `scripts/init_db.py` | ✅ Готов |
| Миграции Alembic | 🔲 Не реализовано |
| Async/Celery batch | 🔲 Не реализовано |
| Тесты pytest | 🔲 Не реализовано |
| Docker | 🔲 Не реализовано |
