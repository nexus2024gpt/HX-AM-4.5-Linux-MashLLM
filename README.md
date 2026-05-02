
# 🔮 HX-AM Proxy v4.2

**Dual-LLM система + 4D-граф формализации + MathCore вычислительный движок.**

Groq генерирует 4D-гипотезы → Gemini верифицирует через трансляцию + стресс-тест → Invariant Engine строит граф с 4D-весами → Archivist оценивает новизну → MathCore подтверждает устойчивость математически.

> Репозиторий v4.0: https://github.com/nexus2024gpt/HX-AM-Proxy-v4-Dual-LLM  
> Репозиторий v4.2: https://github.com/nexus2024gpt/HX-AM-Proxy-v4.2-Dual-LLM-4Dgraf-MathCore

---

## Что нового в v4.2

| Компонент | v4.0 | v4.2 |
|-----------|------|------|
| Генератор | hypothesis + mechanism + b_sync | + `four_d_matrix` (12 числовых параметров) |
| Верификатор | Step 0 (трансляция) | + Step 1 (`stress_test`: tau×1.5, η+0.15) |
| InvariantEngine | text cosine similarity | + `4d_resonance` на рёбрах, `four_d_vec` в узлах |
| **MathCore** | — | KuramotoSimulator · StressTester · ResonanceMatcher · ProbabilityEngine |
| Артефакт | gen, ver, structural, archivist | + `four_d_matrix`, `simulation`, `resonance` |
| API | /query, /graph, /artifacts | + `/math/stress/{id}`, `/insights/feed`, `/math/stats` |
| Миграция | — | `tools/migrate_to_v42.py` для старых артефактов |

---

## 4D-Матрица формализации

| Слой | Параметры | Описание |
|------|-----------|----------|
| **Структура** | C, k, D | Кластеризация, степень узла, фрактальная размерность |
| **Факторы** | h, T, η | Внешнее поле, температура, уровень шума |
| **Динамика** | ω_i, K, K_c, p | Частоты, связь, критический порог, перколяция |
| **Время** | τ, H, freq | Лаг, показатель Херста, частота циклов |

Поддерживаемые модели: `kuramoto` · `percolation` · `ising` · `delay` · `lotka_volterra` · `graph_invariant`

---

## Архитектура

```
[QuestionGenerator] ← Mode A (novel) | Mode B (clarify) | ручной ввод
        │
[PipelineGuard]     ← валидация RAW + нормализация (incl. four_d_matrix)
        │
  Generator (Groq)              Verifier (Gemini)
  → hypothesis                  → Step 0: трансляция
  → four_d_matrix               → Step 1: stress_test
        │                              │
        └──────────┬───────────────────┘
                   ▼
          [Invariant Engine v4.2]
          SemanticSpace  ← text embeddings + 4D vectors
          InvariantGraph ← рёбра с four_d_resonance-бустом
          PhaseDetector  ← фазовые переходы
                   │
          [Archivist]    ← PHENOMENAL | NOVEL | KNOWN | REPHRASING
                   │
          [MathCore]     ← Mode 1: StressTest (λ_max, stability_score)
                   │        Mode 2: ResonanceMatcher (P(A→B))
          artifacts/ + four_d_index.jsonl + sim_results/ + insights/
```

### Формула веса ребра (v4.2)

```
weight = similarity × (1 + domain_distance) × specificity × (1 + four_d_resonance × 0.2)
```

`four_d_resonance` — нормализованный [0,1] изоморфизм 4D-векторов двух узлов.  
Максимальный буст: +20% к весу при полном 4D-совпадении кросс-доменных узлов.

### MathCore — алгоритм стресс-теста

```
1. Базовая симуляция (Kuramoto N=100, t=50·τ)  → r_final, r_mean
2. Stress τ: τ×1.5, τ×2.0                      → stable?
3. Stress η: η+0.15, η+0.30                    → stable?
4. Stress K: K×0.70, K×0.85                   → stable?
5. λ_max = Lyapunov (детерминированный)       → < 0 = устойчиво
6. stability_score = 0.3·base + 0.7·(passed/total)
```

---

## Структура проекта

```
├── hxam_v_4_server.py          — FastAPI + оркестрация
├── invariant_engine.py         — v4.2: SemanticSpace + InvariantGraph + PhaseDetector
├── math_core.py                — NEW: MathCore (StressTester + ResonanceMatcher)
├── llm_client_v_4.py           — LLM клиент (без изменений)
├── archivist.py                — Archivist (без изменений)
├── pipeline_guard.py           — PipelineGuard (без изменений)
├── question_generator.py       — QuestionGenerator (без изменений)
├── response_normalizer.py      — + normalize_four_d_matrix + normalize_stress_test
├── response_normalizer_v42_patch.py — 4D нормализация (патч)
├── api_usage_tracker.py        — APIUsageTracker (без изменений)
├── index_v_4.html              — UI (+ вкладка 📐 Math)
├── hxam_v42_server_patch.py    — Патч-блоки для server.py
├── install_and_run_v42.ps1     — Установка + тест MathCore
├── requirements_v42.txt        — + nolds
├── schemas/
│   ├── __init__.py
│   └── four_d_matrix.py        — NEW: FourDMatrix, compute_4d_resonance
├── prompts/
│   ├── generator_prompt.txt    — v4.2: + four_d_matrix в OUTPUT FORMAT
│   ├── verifier_prompt.txt     — v4.2: + Step 1 stress_test
│   ├── archivist_prompt.txt    — без изменений
│   └── question_generator_prompt.txt — без изменений
├── tools/
│   ├── run_archivist.ps1       — без изменений
│   └── migrate_to_v42.py       — NEW: миграция артефактов v4.0 → v4.2
├── artifacts/
│   ├── semantic_index.jsonl    — text embeddings (авто)
│   ├── four_d_index.jsonl      — NEW: 4D vectors (авто)
│   └── invariant_graph.json    — граф (авто)
├── sim_results/                — NEW: результаты стресс-тестов
├── insights/                   — NEW: вероятностные инсайты
└── docs/
    └── SESSION_CONTEXT.md
```

---

## Установка

```powershell
git clone https://github.com/nexus2024gpt/HX-AM-Proxy-v4.2-Dual-LLM-4Dgraf-MathCore
cd "HX-AM-Proxy-v4.2-Dual-LLM-4Dgraf-MathCore"
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\install_and_run_v42.ps1
```

---

## Миграция архива из v4.0

```powershell
# Предварительный просмотр
python tools/migrate_to_v42.py --dry-run

# Мигрировать все артефакты (вызов LLM для каждого)
python tools/migrate_to_v42.py --delay 1.5

# Мигрировать + стресс-тест каждого
python tools/migrate_to_v42.py --stress

# Один артефакт
python tools/migrate_to_v42.py --id 57cfa5baa346
```

---

## Новые API эндпоинты

| Метод | Путь | Описание |
|-------|------|----------|
| `POST` | `/math/stress/{id}` | Запуск MathCore стресс-теста |
| `GET` | `/insights/feed` | Список вероятностных инсайтов |
| `GET` | `/math/stats` | Статистика MathCore |

---

## Производительность (i5-6300U / 8 GB)

| Операция | Время |
|----------|-------|
| Kuramoto N=100, 50τ | ~0.3 сек |
| Полный стресс-тест | ~0.7 сек |
| 4D-поиск (50 узлов) | <0.1 сек |
| Весь pipeline с MathCore | ~5–7 сек |

---

## Стек

```
FastAPI + Uvicorn
Groq API (llama-3.3-70b-versatile)  — генератор
Gemini API (gemini-2.5-flash)       — верификатор
OpenRouter / HuggingFace             — резервные провайдеры
sentence-transformers (all-MiniLM-L6-v2) — локальные эмбеддинги
scipy.integrate (RK45)               — ODE симуляция Kuramoto
networkx                             — граф + перколяция
nolds                                — показатель Херста (опционально)
numpy / scipy                        — матричные операции
3d-force-graph (WebGL)               — интерактивный 3D граф
```

Персистентность — только flat files (без баз данных):
- `artifacts/semantic_index.jsonl` — text embeddings  
- `artifacts/four_d_index.jsonl` — 4D vectors  
- `artifacts/invariant_graph.json` — граф  
- `sim_results/*.json` — результаты симуляций  
- `insights/*.json` — вероятностные инсайты
