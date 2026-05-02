# HX-AM v4 → v4.2 Migration Plan
## 4D-граф формализации + MathCore

> Репозиторий: https://github.com/nexus2024gpt/HX-AM-Proxy-v4.2-Dual-LLM-4Dgraf-MathCore  
> Среда: i5-6300U · 8 GB RAM · Windows 10 · PowerShell 7.6

---

## Принципы миграции

- **Аддитивный подход**: v4.0 пайплайн не ломается. 4D и MathCore — дополнительные слои.
- **Опциональность**: если 4D-поля отсутствуют в ответе LLM, система работает в режиме v4.0.
- **Изоляция**: `math_core.py` вызывается отдельным процессом/вызовом после `process_with_invariants`.
- **Сохранность данных**: существующие артефакты мигрируются скриптом, оригиналы не удаляются.

---

## Что добавляется в v4.2

| Компонент | v4.0 | v4.2 (новое) |
|-----------|------|-------------|
| Генератор | `hypothesis, mechanism, domain, b_sync, implication` | + `four_d_matrix` (12 параметров) |
| Верификатор | `translation (Step 0), verdict, operationalization` | + `stress_test` (stress_dynamics_stable, tau_robustness) |
| InvariantEngine | text-эмбеддинги + cosine sim | + 4D-вектор (12-dim normalized) + 4d_resonance |
| SemanticSpace | `semantic_index.jsonl` | + `four_d_index.jsonl` |
| **MathCore** | — | `math_core.py`: Mode 1 (StressTest) + Mode 2 (ResonanceMatcher) |
| Артефакт | `gen, ver, structural, archivist` | + `four_d_matrix, stress_test, simulation, prediction` |
| API | `/query, /graph, /artifacts` | + `/math/verify, `/math/stress/{id}`, `/insights/feed` |
| UI | 5 вкладок | + вкладка **📐 Math** |

---

## Этапы миграции (пошагово)

---

### ЭТАП 0 — Инициализация нового проекта

```powershell
# В папке нового проекта D:\Projects\HX-AM-Proxy-v4.2-Dual-LLM-4Dgraf-MathCore

# 1. Скопировать все файлы из v4 (НЕ git clone, копирование)
xcopy "D:\Projects\HX-AM Proxy v4\*" "." /E /H /Y

# 2. Удалить файлы которые будут полностью переписаны
del prompts\generator_prompt.txt
del prompts\verifier_prompt.txt

# 3. Создать новые папки
mkdir schemas
mkdir sim_results
mkdir insights
mkdir archive_v42

# 4. Установить зависимости из requirements_v42.txt
pip install -r requirements_v42.txt
```

**Файлы для копирования без изменений:**
- `llm_client_v_4.py` ✅
- `pipeline_guard.py` ✅  
- `api_usage_tracker.py` ✅
- `archivist.py` (обновить позже на Этапе 6)
- `question_generator.py` ✅
- `index_v_4.html` (обновить позже на Этапе 7)

---

### ЭТАП 1 — Зависимости

**Файл:** `requirements_v42.txt`

Добавляются: `nolds` (Hurst exponent), `numba` опционально.  
scipy, numpy — уже есть, обновить до актуальных версий.

---

### ЭТАП 2 — 4D Schema

**Новый файл:** `schemas/four_d_matrix.py`

Pydantic-модели для 12-параметрического 4D-вектора:
- `FourDStructure` (C, k, D)
- `FourDInfluence` (h, T, eta)
- `FourDDynamics` (omega_i, K, K_c, p)
- `FourDTime` (tau, H, freq)
- `FourDMatrix` — агрегат

Валидация: все float, диапазон [0, 10], защита от LLM-галлюцинаций.

---

### ЭТАП 3 — Generator Prompt

**Файл:** `prompts/generator_prompt.txt`

Добавляем обязательный блок `four_d_matrix` в OUTPUT FORMAT.  
Генератор обязан вернуть 12 числовых параметров + выбрать доминирующую динамическую модель.

---

### ЭТАП 4 — Verifier Prompt

**Файл:** `prompts/verifier_prompt.txt`

Добавляем Step 1 (Stress-Test):
- После трансляции (Step 0) верификатор обязан указать:
  - `stress_dynamics_stable` (bool): выживает ли K при τ × 1.5 + η + 0.15
  - `tau_robustness` (float): максимальный τ до потери когерентности

---

### ЭТАП 5 — response_normalizer.py (обновление)

**Файл:** `response_normalizer.py`

Добавить:
- `normalize_four_d_matrix()` — нормализация 4D из LLM
- Псевдонимы полей 4D на русском/английском  
- Дефолты при отсутствии 4D (`four_d_matrix: null`)

---

### ЭТАП 6 — MathCore

**Новый файл:** `math_core.py`

Ключевые классы:
- `ModelRegistry` — маппинг доминирующей модели по динамике
- `KuramotoSimulator` — ODE для Kuramoto (scipy.integrate.solve_ivp)
- `PercolationSimulator` — Monte-Carlo перколяция
- `StressTester` — Mode 1: расшатывание параметров + λ_max
- `ResonanceMatcher` — Mode 2: 4D-поиск в архиве
- `ProbabilityEngine` — вычисление P(A→B)
- `MathCore` — оркестратор

**Ограничения под железо:**
- N_oscillators ≤ 200 (RAM)
- t_max = 50·τ (не 100·τ как в симуляции)
- Без numba (overhead компиляции)
- Кэш симуляций: `sim_results/{id}_cache.npz`

---

### ЭТАП 7 — invariant_engine.py (обновление)

**Файл:** `invariant_engine_v42.py` (→ заменит `invariant_engine.py`)

Добавить:
- `FourDSemanticSpace` — хранит 4D-векторы отдельно от text-embeddings
- `compute_4d_distance()` — взвешенное евклидово расстояние по 4 слоям
- `compute_4d_resonance()` — нормализованный [0,1] показатель изоморфизма
- В `process_with_invariants()` — вычислять `4d_resonance` если есть `four_d_matrix`

---

### ЭТАП 8 — hxam_v_4_server.py (обновление)

Добавить эндпоинты:
- `POST /math/stress/{artifact_id}` — запуск StressTest для артефакта
- `GET /insights/feed` — список инсайтов
- `GET /math/stats` — статистика MathCore симуляций

Изменить `process_query()`:
- После `process_with_invariants()` вызывать `MathCore.stress_test()` если `four_d_matrix` есть
- Записывать `simulation` блок в артефакт

---

### ЭТАП 9 — Миграция архива артефактов

**Файл:** `tools/migrate_to_v42.py`

CLI-скрипт:
```powershell
python tools/migrate_to_v42.py --dry-run   # показать список
python tools/migrate_to_v42.py             # мигрировать всё
python tools/migrate_to_v42.py --id 57cfa5baa346  # один артефакт
```

Что делает:
1. Читает существующий артефакт
2. Вызывает LLM для извлечения 4D-матрицы из `hypothesis + mechanism`
3. Нормализует, валидирует
4. Добавляет `four_d_matrix` в артефакт
5. Обновляет `four_d_index.jsonl`

---

### ЭТАП 10 — UI обновление (index_v_4.html)

Добавить вкладку **📐 Math**:
- Кнопка "Запустить Stress-Test" для текущего артефакта
- Вывод: stability_score, λ_max, границы устойчивости
- Список инсайтов из `/insights/feed`
- 4D-матрица текущего результата

---

## Порядок Git-коммитов

```
feat: init v4.2 project structure
feat: add 4D matrix schema (schemas/four_d_matrix.py)
feat: update generator prompt with four_d_matrix requirement
feat: update verifier prompt with stress_test step
feat: add response_normalizer 4D normalization
feat: add MathCore module (math_core.py) - Mode 1 StressTest
feat: add MathCore Mode 2 ResonanceMatcher
feat: extend InvariantEngine with FourDSemanticSpace
feat: add math endpoints to server (/math/stress, /insights)
feat: add Math tab to UI
feat: add artifact migration script
docs: update README for v4.2
```

---

## Проверка после каждого этапа

```powershell
# После Этапа 2 — валидация схемы
python -c "from schemas.four_d_matrix import FourDMatrix; print('OK')"

# После Этапа 6 — тест MathCore
python math_core.py --test

# После Этапа 7 — тест 4D-поиска
python invariant_engine.py --test-4d

# После Этапа 8 — запуск сервера
python hxam_v_4_server.py
# → открыть http://localhost:8000 → вкладка 📐 Math
```

---

## Откат (при проблемах)

Все новые поля `four_d_matrix`, `stress_test`, `simulation` — **опциональны** в артефакте.  
Если MathCore упал, пайплайн продолжается без него (try/except в server).  
Старые артефакты без 4D работают в режиме v4.0.

---

## Расчёт производительности (i5-6300U)

| Операция | Время | Условия |
|----------|-------|---------|
| Kuramoto N=100, 50τ | ~0.3 сек | scipy.solve_ivp |
| Stress-test (5 уровней) | ~2 сек | N=100, без numba |
| Percolation N=1000, MC×20 | ~1.5 сек | networkx |
| 4D-поиск в архиве (50 узлов) | <0.1 сек | numpy cosine |
| Полный цикл с MathCore | ~5-7 сек | vs ~3-4 сек без MathCore |
