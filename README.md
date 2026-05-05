```markdown
# 🔮 HX-AM Proxy v4.5

**Dual-LLM система + 4D-граф формализации + MathCore вычислительный движок + Децентрализованный AI**

mesh-llm генерирует 4D-гипотезы через публичную P2P-сеть → Gemini/Mesh верифицирует через трансляцию + стресс-тест → Invariant Engine строит граф с 4D-весами → Archivist оценивает новизну → MathCore подтверждает устойчивость математически.

> Репозиторий v4.0: https://github.com/nexus2024gpt/HX-AM-Proxy-v4-Dual-LLM  
> Репозиторий v4.2: https://github.com/nexus2024gpt/HX-AM-Proxy-v4.2-Dual-LLM-4Dgraf-MathCore  
> **Репозиторий v4.5**: https://github.com/nexus2024gpt/HX-AM-4.5-Linux-MashLLM

---

## Что нового в v4.5

| Компонент | v4.2 | v4.5 |
|-----------|------|------|
| Генератор | Groq (llama-3.3-70b) | **mesh-llm (Qwen3-4B / 8B / 9B)** + P2P-сеть |
| Верификатор | Gemini (gemini-2.5-flash) | mesh-llm + Gemini + OpenRouter (fallback) |
| Провайдерная логика | Прямые вызовы API | **APIUsageTracker** с приоритетами и account-level ротацией |
| Локальный LLM | — | **mesh-llm** (CPU, 5.8 GB RAM, модели Qwen3) |
| Контекстное окно | 8k | **16k–32k** (зависит от выбранной модели) |
| OS | Windows | **Windows + WSL2 (Ubuntu 24.04)** |
| Промпты | v4.2 (12 параметров) | без изменений |
| MathCore | Kuramoto, Ising, LV, etc. | без изменений |

**Доступные модели mesh-llm (все скачаны локально):**
- `Qwen3.5-9B-Q4_K_M` (5.3 GB, приоритет 1)
- `Qwen3-8B-Q4_K_M` (4.7 GB, приоритет 2)
- `Qwen3-4B-Q4_K_M` (2.4 GB, приоритет 3)

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

## Архитектура (v4.5)

```
[User Input] → Mode A (novel) | Mode B (clarify) | ручной ввод
        │
[APIUsageTracker]  ← приоритеты: mesh (p1-3) → Groq (p6-7) → Gemini (p3-7) → OpenRouter (p8-9)
        │
  Generator (mesh-llm)         Verifier (mesh-llm / Gemini)
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

### Цепочка провайдеров (генератор)
```
mesh-llm (Qwen3.5-9B, p1) → mesh-llm (Qwen3-8B, p2) → mesh-llm (Qwen3-4B, p3) 
→ Groq Nexus (p6) → Groq Roman (p7)
```

Верификатор аналогично начинает с mesh-llm и далее переходит на Gemini/OpenRouter.

---

## Установка (WSL2 / Linux)

```bash
# Клонирование репозитория
cd ~
git clone https://github.com/nexus2024gpt/HX-AM-4.5-Linux-MashLLM.git hxam
cd hxam

# Настройка виртуального окружения Python 3.11
python3.11 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements_v42.txt
```

---

## Запуск mesh-llm (обязательный компонент)

Выберите модель из трёх доступных (в порядке убывания качества и требований к памяти):

```bash
# Самая мощная (9B, ~6.5 GB RAM)
mesh-llm serve --model /home/roman220877/.cache/huggingface/hub/models--unsloth--Qwen3.5-9B-GGUF/Qwen3.5-9B-Q4_K_M.gguf --ctx-size 32768

# Сбалансированная (8B, ~5.5 GB RAM)
mesh-llm serve --model Qwen3-8B-Q4_K_M --ctx-size 16384

# Самая лёгкая (4B, ~3.5 GB RAM)
mesh-llm serve --model Qwen3-4B-Q4_K_M --ctx-size 16384
```

API будет доступен на `http://localhost:9337/v1`.

---

## Запуск HX-AM

```bash
cd ~/hxam
source venv/bin/activate
python hxam_v_4_server.py
```

Открыть в браузере: `http://localhost:8000`

---

## Стек (v4.5)

```
FastAPI + Uvicorn
mesh-llm (Qwen3-8B-Q4_K_M)           — основной генератор
Gemini API (gemini-2.5-flash)         — верификатор (резерв)
Groq API (llama-3.3-70b-versatile)   — резервный генератор
OpenRouter / HuggingFace              — последний рубеж
sentence-transformers (all-MiniLM-L6-v2) — локальные эмбеддинги
scipy.integrate (RK45)                — ODE симуляция Kuramoto
networkx                              — граф + перколяция
nolds                                 — показатель Херста (опционально)
numpy / scipy                         — матричные операции
```

---

## Ссылки

- mesh-llm: https://github.com/Mesh-LLM/mesh-llm
- HX-AM v4.0: https://github.com/nexus2024gpt/HX-AM-Proxy-v4-Dual-LLM
- HX-AM v4.2: https://github.com/nexus2024gpt/HX-AM-Proxy-v4.2-Dual-LLM-4Dgraf-MathCore
- **HX-AM v4.5**: https://github.com/nexus2024gpt/HX-AM-4.5-Linux-MashLLM
```