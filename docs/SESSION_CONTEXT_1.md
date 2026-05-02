# HX-AM Proxy v4.2 — Контекст сессии разработки
# Формат: справочный документ для передачи между сессиями
# Дата: апрель 2026
# Репозиторий: https://github.com/nexus2024gpt/HX-AM-Proxy-v4.2-Dual-LLM-4Dgraf-MathCore

---

## Что изменилось от v4.0 к v4.2

### Философия
v4.0: семантическое пространство (text embeddings) + структурный граф (cosine similarity).  
v4.2: добавляется **математическая верификация** через 4D-матрицу формализации (12 числовых параметров) + **численная симуляция** (Kuramoto ODE) + **вероятностный прогноз** резонансов между доменами.

Ключевой принцип: инвариант теперь не просто «семантически похож» — он **математически выживает** при расшатывании параметров (стресс-тест).

---

## Новые компоненты (v4.2)

### schemas/four_d_matrix.py
- `FourDStructure`, `FourDInfluence`, `FourDDynamics`, `FourDTime` — Pydantic-модели
- `FourDMatrix` — агрегат, метод `.to_vector()` → 13-мерный нормализованный numpy array
- `compute_4d_resonance(a, b)` → float [0,1]: взвешенный изоморфизм
- `compute_4d_distance(a, b)` → float: взвешенное евклидово расстояние
- Веса слоёв: structure=0.25, influence=0.25, dynamics=0.30, time=0.20

### math_core.py
- `KuramotoSimulator`: RK45, N≤200, ИСПРАВЛЕННЫЙ знак diff = θ_j − θ_i
- `StabilityAnalyzer.lyapunov_estimate()`: детерминированный (без шума), прогрев 100 шагов
- `StressTester`: стресс по τ, η, K → stability_score [0,1]
- `ResonanceMatcher`: 4D-поиск в `four_d_index.jsonl` + 60% cosine + 40% euclidean
- `ProbabilityEngine`: P(A→B) = σ(α·Iso + β·Stability + γ·Scale + δ·Survival − ε·Noise)
- `MathCore`: оркестратор, методы `.stress_test()`, `.find_resonance()`, `.index_artifact()`
- CLI: `python math_core.py --test`, `python math_core.py --stress <id>`

**Критический баг исправлен:** в v4.0-версии Kuramoto знак разности фаз был `θ_i − θ_j` вместо `θ_j − θ_i` → антисинхронизация. Исправлено: r_final при K>K_c = 0.9965.

### tools/migrate_to_v42.py
- Читает hypothesis + mechanism → вызывает LLM для извлечения four_d_matrix
- Нормализует через `FourDMatrix.from_raw()`
- Добавляет в артефакт + four_d_index.jsonl
- Флаги: `--dry-run`, `--stress`, `--id`, `--delay`

### response_normalizer_v42_patch.py
- `normalize_four_d_matrix(raw, repairs)` → нормализованный dict или None
- `normalize_stress_test(raw, repairs)` → нормализованный dict или None
- Обрабатывает: вложенные словари, плоскую структуру, псевдонимы на русском

---

## Изменения в существующих файлах

### prompts/generator_prompt.txt
Добавлен блок `four_d_matrix` в REQUIRED fields и OUTPUT FORMAT.  
Критически важно: все 12 параметров обязательны, модель из списка known models.

### prompts/verifier_prompt.txt
Добавлен Step 1 (stress_test) после Step 0 (трансляция).  
Верификатор обязан вернуть: `stress_dynamics_stable` (bool), `tau_robustness` (float), `eta_critical` (float).

### invariant_engine.py
- `SemanticSpace.add()` принимает `four_d_vec` (опционально)
- `SemanticSpace.four_d_vec_by_id()`, `store_four_d_vec()` — новые методы
- `InvariantGraph.add_edge()` принимает `four_d_resonance` → буст к весу (+20% max)
- `process_with_invariants()` вычисляет `top_4d_resonance`, `has_four_d`, `stress_stable`
- Узлы в графе: дополнительные атрибуты `has_four_d`, `stress_stable`

---

## Производительность и ограничения

```python
N_OSCILLATORS_MAX = 200   # Kuramoto агентов (RAM limit)
T_MAX_FACTOR = 50         # t_max = 50 · τ (CPU limit)
STRESS_LEVELS = [0.1, 0.3, 0.5]  # три уровня расшатывания
# Полный стресс-тест: ~0.7–1.5 сек на i5-6300U
# Полный pipeline с MathCore: ~5–7 сек vs ~3–4 сек без MathCore
```

Lyapunov: детерминированный estimator (η=0), прогрев 100 шагов, 300 шагов оценки.  
При K > K_c (синхронизированная система) λ_max < 0 (асимптотически устойчиво).

---

## Новые персистентные файлы

| Файл | Формат | Содержимое |
|------|--------|------------|
| `artifacts/four_d_index.jsonl` | JSONL | id, domain, vector (13-dim), stability_score |
| `sim_results/{id}_stress.json` | JSON | Полный отчёт StressTester |
| `insights/{id}.json` | JSON | Вероятностный инсайт из ResonanceMatcher |

---

## Порядок интеграции (TODO для следующей сессии)

1. **Применить патч в response_normalizer.py** (из response_normalizer_v42_patch.py)  
   Добавить вызовы normalize_four_d_matrix() в normalize_gen(), normalize_stress_test() в normalize_ver().

2. **Применить патч в hxam_v_4_server.py** (из hxam_v42_server_patch.py)  
   - Блок 1: импорты MathCore
   - Блок 2: инициализация math_core singleton
   - Блок 5: вызов MathCore в process_query() после archivist
   - Блок 6: новые эндпоинты (/math/stress, /insights/feed, /math/stats)

3. **Добавить вкладку 📐 Math в index_v_4.html**  
   - Кнопка запуска stress_test
   - Отображение four_d_matrix текущего результата
   - Лента insights из /insights/feed

4. **Запустить миграцию архива**
   ```powershell
   python tools/migrate_to_v42.py --dry-run
   python tools/migrate_to_v42.py --delay 1.5
   ```

5. **Git push**
   ```powershell
   git add .
   git commit -m "feat: complete v4.2 migration (4D matrix + MathCore)"
   git push origin main
   ```

---

## Переменные окружения (.env)
Без изменений от v4.0. Новых ключей не требуется.

---

## Известные проблемы v4.2

| Проблема | Статус |
|----------|--------|
| LLM не всегда возвращает four_d_matrix | Нормализатор ставит None, pipeline продолжается в режиме v4.0 |
| Lyapunov > 0 при высоком η | Исправлено: estimator детерминированный (η=0) |
| Kuramoto антисинхронизация | Исправлено: diff = θ_j − θ_i (sign fix) |
| migrate_to_v42.py требует LLM-ключи | Ожидаемо; без ключей — только --dry-run |

---

## Математические принципы (краткая справка)

**Kuramoto:** r(t) = |Σ exp(iθ_j)| / N → 1 при полной синхронизации  
**Критический порог:** K_c ≈ 2γ/g(0) (для Лоренц-распределения частот)  
**λ_max < 0** → малые возмущения затухают → устойчивый аттрактор  
**P(A→B) = σ(α·Iso_4D + β·Stability + γ·Scale + δ·Survival − ε·Noise)**  
Логистическая калибровка: k=5.0, x0=0.60; P > 0.75 = "high", > 0.55 = "plausible"
