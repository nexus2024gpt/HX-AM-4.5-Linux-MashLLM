# schemas/four_d_matrix.py — HX-AM v4.2
"""
Pydantic-схемы для 4D-матрицы формализации.

4D-матрица описывает инвариант по четырём ортогональным слоям:
  Структура  (Topology)   — геометрия и связи системы
  Факторы    (Influence)  — внешнее давление и шум
  Динамика   (Dynamics)   — процесс обмена/движения
  Время      (Temporal)   — ритм и задержки

Все параметры — float в диапазоне [0, 10] (защита от LLM-галлюцинаций).
Используется для:
  - Валидации вывода Generator
  - Построения 12-мерного 4D-вектора
  - Поиска изоморфных артефактов
  - Запуска MathCore симуляций
"""

from __future__ import annotations

from typing import Any, Dict, Optional
import math

try:
    from pydantic import BaseModel, Field, field_validator, model_validator
    PYDANTIC_V2 = True
except ImportError:
    from pydantic import BaseModel, Field, validator
    PYDANTIC_V2 = False

import numpy as np

# ──────────────────────────────────────────────
# КОНСТАНТЫ
# ──────────────────────────────────────────────

# Параметры по слоям для нормализации (max значение для MinMax-нормализации)
LAYER_RANGES = {
    # Структура
    "C":      (0.0, 1.0),   # коэффициент кластеризации
    "k":      (1.0, 50.0),  # средняя степень узла
    "D":      (1.0, 4.0),   # фрактальная размерность

    # Факторы
    "h":      (0.0, 5.0),   # напряжённость внешнего поля
    "T":      (0.0, 5.0),   # температура (аналог случайности)
    "eta":    (0.0, 1.0),   # уровень шума η

    # Динамика
    "omega_i": (0.0, 5.0),  # собственная частота осцилляторов
    "K":       (0.0, 2.0),  # константа связи
    "K_c":     (0.0, 2.0),  # критическая константа связи
    "p":       (0.0, 1.0),  # вероятность протекания / связности

    # Время
    "tau":     (0.0, 20.0), # лаг/задержка
    "H":       (0.0, 1.0),  # показатель Херста
    "freq":    (0.0, 10.0), # характерная частота циклов
}

# Веса слоёв для 4D-расстояния (приоритет динамике)
LAYER_WEIGHTS = {
    "structure": 0.25,
    "influence": 0.25,
    "dynamics":  0.30,
    "time":      0.20,
}

# Карта доминирующих математических моделей
MODEL_REGISTRY: Dict[str, str] = {
    "kuramoto":         "Kuramoto(ω_i, K, K_c)",
    "percolation":      "Percolation(p, p_c, giant_component)",
    "lotka_volterra":   "Lotka-Volterra(α, β, γ, δ)",
    "ising":            "Ising(h, T, J)",
    "delay":            "Delay-System(τ, H, 1/f)",
    "graph_invariant":  "Graph_Invariants(C, k, D)",
    "fram":             "FRAM(preconditions, resources, timing, control)",
    "coleman":          "Coleman(ΔE, phase_lock)",
}


def _clamp(v: Any, lo: float = 0.0, hi: float = 10.0) -> float:
    """Принудительно приводит значение к float и зажимает в диапазон."""
    try:
        f = float(str(v).strip().replace(",", "."))
        if math.isnan(f) or math.isinf(f):
            return (lo + hi) / 2
        return max(lo, min(hi, f))
    except (ValueError, TypeError):
        return (lo + hi) / 2


# ──────────────────────────────────────────────
# LAYER MODELS
# ──────────────────────────────────────────────

class FourDStructure(BaseModel):
    """Топология системы."""
    C:  float = Field(default=0.5, ge=0.0, le=1.0,  description="Коэффициент кластеризации")
    k:  float = Field(default=6.0, ge=1.0, le=50.0, description="Средняя степень узла")
    D:  float = Field(default=2.0, ge=1.0, le=4.0,  description="Фрактальная размерность")

    model_config = {"extra": "ignore"}

    @classmethod
    def from_raw(cls, data: Dict[str, Any]) -> "FourDStructure":
        return cls(
            C=_clamp(data.get("C", 0.5),  0.0, 1.0),
            k=_clamp(data.get("k", 6.0),  1.0, 50.0),
            D=_clamp(data.get("D", 2.0),  1.0, 4.0),
        )


class FourDInfluence(BaseModel):
    """Внешнее давление и шум."""
    h:   float = Field(default=0.5, ge=0.0, le=5.0, description="Напряжённость внешнего поля")
    T:   float = Field(default=1.0, ge=0.0, le=5.0, description="Температура / стохастичность")
    eta: float = Field(default=0.2, ge=0.0, le=1.0, description="Уровень шума η")

    model_config = {"extra": "ignore"}

    @classmethod
    def from_raw(cls, data: Dict[str, Any]) -> "FourDInfluence":
        return cls(
            h=_clamp(data.get("h",   0.5), 0.0, 5.0),
            T=_clamp(data.get("T",   1.0), 0.0, 5.0),
            eta=_clamp(data.get("eta", 0.2), 0.0, 1.0),
        )


class FourDDynamics(BaseModel):
    """Процесс обмена и движения."""
    omega_i: float = Field(default=0.25, ge=0.0, le=5.0,  description="Собственная частота осцилляторов")
    K:       float = Field(default=0.35, ge=0.0, le=2.0,  description="Константа связи")
    K_c:     float = Field(default=0.48, ge=0.0, le=2.0,  description="Критическая константа связи")
    p:       float = Field(default=0.65, ge=0.0, le=1.0,  description="Вероятность протекания")
    model:   str   = Field(default="kuramoto", description="Доминирующая мат. модель")

    model_config = {"extra": "ignore"}

    @classmethod
    def from_raw(cls, data: Dict[str, Any]) -> "FourDDynamics":
        return cls(
            omega_i=_clamp(data.get("omega_i", 0.25), 0.0, 5.0),
            K=_clamp(data.get("K",   0.35), 0.0, 2.0),
            K_c=_clamp(data.get("K_c", 0.48), 0.0, 2.0),
            p=_clamp(data.get("p",   0.65), 0.0, 1.0),
            model=str(data.get("model", "kuramoto")).lower().strip(),
        )

    @property
    def is_above_critical(self) -> bool:
        """K > K_c → система за порогом синхронизации."""
        return self.K > self.K_c


class FourDTime(BaseModel):
    """Временнáя структура."""
    tau:  float = Field(default=0.5,  ge=0.0, le=20.0, description="Характерный лаг/задержка")
    H:    float = Field(default=0.7,  ge=0.0, le=1.0,  description="Показатель Херста")
    freq: float = Field(default=1.0,  ge=0.0, le=10.0, description="Частота циклов ω")

    model_config = {"extra": "ignore"}

    @classmethod
    def from_raw(cls, data: Dict[str, Any]) -> "FourDTime":
        return cls(
            tau=_clamp(data.get("tau",  0.5),  0.0, 20.0),
            H=_clamp(data.get("H",    0.7),  0.0, 1.0),
            freq=_clamp(data.get("freq", 1.0),  0.0, 10.0),
        )


# ──────────────────────────────────────────────
# АГРЕГАТ
# ──────────────────────────────────────────────

class FourDMatrix(BaseModel):
    """Полная 4D-матрица инварианта (12+ параметров)."""
    structure: FourDStructure
    influence: FourDInfluence
    dynamics:  FourDDynamics
    time:      FourDTime

    model_config = {"extra": "ignore"}

    @classmethod
    def from_raw(cls, data: Dict[str, Any]) -> Optional["FourDMatrix"]:
        """
        Создаёт FourDMatrix из сырого LLM-словаря.
        Обрабатывает: вложенные ключи, плоскую структуру, псевдонимы.
        Возвращает None если данных недостаточно.
        """
        if not data:
            return None

        # Пробуем вложенную структуру
        s_raw = data.get("structure") or data.get("структура") or {}
        i_raw = data.get("influence") or data.get("факторы") or {}
        d_raw = data.get("dynamics")  or data.get("динамика") or {}
        t_raw = data.get("time")      or data.get("время")    or {}

        # Если вложенных нет — ищем плоско
        if not any([s_raw, i_raw, d_raw, t_raw]):
            s_raw = {k: data[k] for k in ("C", "k", "D") if k in data}
            i_raw = {k: data[k] for k in ("h", "T", "eta") if k in data}
            d_raw = {k: data[k] for k in ("omega_i", "K", "K_c", "p", "model") if k in data}
            t_raw = {k: data[k] for k in ("tau", "H", "freq") if k in data}

        if not any([s_raw, d_raw, t_raw]):
            return None

        try:
            return cls(
                structure=FourDStructure.from_raw(s_raw if isinstance(s_raw, dict) else {}),
                influence=FourDInfluence.from_raw(i_raw if isinstance(i_raw, dict) else {}),
                dynamics=FourDDynamics.from_raw(d_raw if isinstance(d_raw, dict) else {}),
                time=FourDTime.from_raw(t_raw if isinstance(t_raw, dict) else {}),
            )
        except Exception:
            return None

    def to_vector(self) -> np.ndarray:
        """
        Возвращает нормализованный 12-мерный вектор [0, 1].
        Порядок: C, k, D | h, T, eta | omega_i, K, K_c, p | tau, H, freq
        """
        raw = np.array([
            self.structure.C, self.structure.k, self.structure.D,
            self.influence.h, self.influence.T, self.influence.eta,
            self.dynamics.omega_i, self.dynamics.K, self.dynamics.K_c, self.dynamics.p,
            self.time.tau, self.time.H, self.time.freq,
        ], dtype=np.float64)

        ranges = [
            LAYER_RANGES["C"],  LAYER_RANGES["k"],  LAYER_RANGES["D"],
            LAYER_RANGES["h"],  LAYER_RANGES["T"],  LAYER_RANGES["eta"],
            LAYER_RANGES["omega_i"], LAYER_RANGES["K"], LAYER_RANGES["K_c"], LAYER_RANGES["p"],
            LAYER_RANGES["tau"], LAYER_RANGES["H"], LAYER_RANGES["freq"],
        ]
        lo = np.array([r[0] for r in ranges])
        hi = np.array([r[1] for r in ranges])
        span = hi - lo
        span[span == 0] = 1e-9
        return np.clip((raw - lo) / span, 0.0, 1.0)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "structure": {"C": self.structure.C, "k": self.structure.k, "D": self.structure.D},
            "influence": {"h": self.influence.h, "T": self.influence.T, "eta": self.influence.eta},
            "dynamics":  {
                "omega_i": self.dynamics.omega_i, "K": self.dynamics.K,
                "K_c": self.dynamics.K_c, "p": self.dynamics.p, "model": self.dynamics.model,
            },
            "time": {"tau": self.time.tau, "H": self.time.H, "freq": self.time.freq},
        }

    def dominant_model(self) -> str:
        return self.dynamics.model


# ──────────────────────────────────────────────
# УТИЛИТЫ
# ──────────────────────────────────────────────

def compute_4d_distance(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """
    Взвешенное евклидово расстояние в 4D-пространстве.
    Каждый слой взвешивается по LAYER_WEIGHTS.
    Вектора размерности 13 (порядок: C,k,D | h,T,eta | omega_i,K,K_c,p | tau,H,freq).
    """
    if vec_a.shape != vec_b.shape or len(vec_a) != 13:
        diff = vec_a - vec_b
        return float(np.sqrt(np.dot(diff, diff)))

    # Слои по индексам
    w = LAYER_WEIGHTS
    slices = [
        (slice(0, 3),  w["structure"]),
        (slice(3, 6),  w["influence"]),
        (slice(6, 10), w["dynamics"]),
        (slice(10, 13), w["time"]),
    ]
    total = 0.0
    for sl, weight in slices:
        diff = vec_a[sl] - vec_b[sl]
        total += weight * float(np.dot(diff, diff))
    return float(np.sqrt(total))


def compute_4d_resonance(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """Нормализованная мера изоморфизма [0, 1]. 1 = идеальное совпадение."""
    dist = compute_4d_distance(vec_a, vec_b)
    # Максимально возможное расстояние при весах ≤ 1 и вектор в [0,1]^13
    max_dist = float(np.sqrt(sum(w for w in LAYER_WEIGHTS.values())))
    return round(max(0.0, 1.0 - dist / max(max_dist, 1e-9)), 3)
