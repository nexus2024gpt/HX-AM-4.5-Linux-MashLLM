# mgap_lib/engine/topology_validator.py — HX-AM v4.7
"""
Топологический фильтр резонанса.

Проблема:
  _compute_resonance() использует косинусный резонанс 13-мерных векторов.
  Физика разломов и дефолты банков могут иметь похожие 4D-векторы,
  но принципиально разные топологии сетей. Жёсткое совпадение
  math_type не достаточно — внутри одного класса (percolation) граф
  Эрдёша-Реньи и безмасштабный граф дают разные критические пороги.

Решение:
  Вместо жёсткого if/else используем НЕПРЕРЫВНЫЕ функции совместимости.
  Каждый параметр (C, k, D) вносит взвешенный вклад в итоговый
  топологический коэффициент penalty ∈ (0, 1].

  penalty = 1.0 означает полную топологическую совместимость.
  penalty = 0.1 означает фундаментальную несовместимость.

  Итоговый резонанс: resonance_adjusted = resonance * penalty

ПРИНЦИП "МЯГКИХ ГРАНИЦ":
  Реальные сети редко являются чисто Scale-Free или чисто ER.
  Переход между режимами описывается сигмоидными функциями,
  а не ступенчатыми if/else — это предотвращает срезание
  межнаучных инвариантов из-за незначительного отклонения
  фрактальной размерности или коэффициента кластеризации.
"""

from __future__ import annotations

import logging
import math
from typing import Dict, Optional, Tuple

logger = logging.getLogger("HXAM.mgap.topology")


def _sigmoid(x: float, center: float, steepness: float) -> float:
    """
    Сигмоидная функция [0, 1].
    x=center → 0.5, steepness контролирует крутизну перехода.
    Высокий steepness → более резкий переход.
    Низкий steepness (например, 2) → плавный переход на ±2σ.
    """
    try:
        return 1.0 / (1.0 + math.exp(-steepness * (x - center)))
    except OverflowError:
        return 1.0 if x > center else 0.0


def _topology_profile(C: float, k: float, D: float) -> Dict[str, float]:
    """
    Вычисляет «профиль топологии» — непрерывные вероятности принадлежности
    к каждому классу топологий. Сумма не равна 1 (нечёткая классификация).

    Топологические признаки:
      Scale-Free:  высокое D (фрактальность), высокое C, умеренное k
      Small-World: умеренное C (0.4–0.7), небольшое k, D~2
      Erdos-Renyi: низкое C, любое k, D~2
      Regular:     очень низкое C, малое k, целочисленное D

    Параметры сигмоид подобраны так, чтобы переходы были плавными
    при отклонении ±0.1 от характерных значений.
    """
    # Scale-Free: D → высокое (>2.3) и C → высокое (>0.55)
    sf_from_D  = _sigmoid(D, center=2.4, steepness=8.0)   # крутой переход по D
    sf_from_C  = _sigmoid(C, center=0.58, steepness=10.0)  # крутой по C
    scale_free = sf_from_D * sf_from_C

    # Small-World: умеренное C (пик ~0.5), k небольшое (<20)
    sw_from_C  = _sigmoid(C, center=0.35, steepness=8.0) * _sigmoid(-C, center=-0.72, steepness=8.0)
    sw_from_k  = _sigmoid(-k, center=-20.0, steepness=0.3)  # плавный спад при k>20
    small_world = sw_from_C * sw_from_k

    # Erdos-Renyi: низкое C (<0.3), произвольное k
    er_from_C  = _sigmoid(-C, center=-0.30, steepness=12.0)  # низкое C
    erdos_renyi = er_from_C

    # Regular: очень низкое C, малое k
    reg_from_C = _sigmoid(-C, center=-0.15, steepness=15.0)
    reg_from_k = _sigmoid(-k, center=-8.0, steepness=0.8)
    regular    = reg_from_C * reg_from_k

    return {
        "scale_free":  round(scale_free,  4),
        "small_world": round(small_world, 4),
        "erdos_renyi": round(erdos_renyi, 4),
        "regular":     round(regular,     4),
    }


# Матрица совместимости топологий.
# Значения — базовые коэффициенты совместимости [0, 1].
# Используются как веса при свёртке профилей.
_COMPAT_MATRIX: Dict[Tuple[str, str], float] = {
    ("scale_free",  "scale_free"):  1.00,
    ("scale_free",  "small_world"): 0.65,
    ("scale_free",  "erdos_renyi"): 0.40,
    ("scale_free",  "regular"):     0.15,

    ("small_world", "scale_free"):  0.65,
    ("small_world", "small_world"): 1.00,
    ("small_world", "erdos_renyi"): 0.72,
    ("small_world", "regular"):     0.45,

    ("erdos_renyi", "scale_free"):  0.40,
    ("erdos_renyi", "small_world"): 0.72,
    ("erdos_renyi", "erdos_renyi"): 1.00,
    ("erdos_renyi", "regular"):     0.55,

    ("regular",     "scale_free"):  0.15,
    ("regular",     "small_world"): 0.45,
    ("regular",     "erdos_renyi"): 0.55,
    ("regular",     "regular"):     1.00,
}

# math_type, для которых топологический штраф НЕ применяется
# (модели агностичны к топологии носителя)
_TOPOLOGY_AGNOSTIC_TYPES = frozenset({"kuramoto", "delay", "ising", "lotka_volterra"})

# Минимальный нижний порог penalty (чтобы не занулять кросс-доменные инварианты)
_MIN_PENALTY = 0.20


class TopologyValidator:
    """
    Вычисляет топологический коэффициент совместимости penalty ∈ [MIN_PENALTY, 1.0].

    Алгоритм:
      1. Для артефакта и модели вычисляем «профили топологии» —
         нечёткое членство в каждом классе (Scale-Free, ER, SW, Regular).
      2. Свёртываем профили через матрицу совместимости.
      3. Итоговый penalty = взвешенная сумма совместимостей.

    Для math_type из _TOPOLOGY_AGNOSTIC_TYPES: penalty = 1.0 (нет штрафа).
    """

    def compute_penalty(
        self,
        art_four_d: Dict,
        model: Dict,
    ) -> Tuple[float, Dict]:
        """
        Вычисляет топологический penalty.

        Returns:
            (penalty: float, debug_info: Dict)
        """
        mt = str(model.get("math_type", "")).lower().strip()

        # Агностичные типы — без штрафа
        if mt in _TOPOLOGY_AGNOSTIC_TYPES:
            return 1.0, {"reason": f"math_type={mt} is topology-agnostic"}

        model_four_d = model.get("four_d_matrix") or {}

        art_s = art_four_d.get("structure", {})
        art_C = float(art_s.get("C", 0.5))
        art_k = float(art_s.get("k", 6.0))
        art_D = float(art_s.get("D", 2.0))

        mod_s = model_four_d.get("structure", {})
        mod_C = float(mod_s.get("C", 0.5))
        mod_k = float(mod_s.get("k", 6.0))
        mod_D = float(mod_s.get("D", 2.0))

        art_profile = _topology_profile(art_C, art_k, art_D)
        mod_profile = _topology_profile(mod_C, mod_k, mod_D)

        # Свёртка профилей через матрицу совместимости
        penalty = 0.0
        total_weight = 0.0
        for art_topo, art_prob in art_profile.items():
            for mod_topo, mod_prob in mod_profile.items():
                weight = art_prob * mod_prob
                compat = _COMPAT_MATRIX.get((art_topo, mod_topo), 0.6)
                penalty      += compat * weight
                total_weight += weight

        if total_weight < 1e-9:
            penalty = 0.6  # нет информации → нейтральный штраф
        else:
            penalty = penalty / total_weight

        # Нижний порог — не занулять кросс-доменные инварианты
        penalty = max(_MIN_PENALTY, round(penalty, 4))

        debug = {
            "art_profile": art_profile,
            "mod_profile": mod_profile,
            "raw_penalty": round(penalty, 4),
            "min_penalty": _MIN_PENALTY,
            "art_structure": {"C": art_C, "k": art_k, "D": art_D},
            "mod_structure": {"C": mod_C, "k": mod_k, "D": mod_D},
        }
        logger.debug(
            f"TopologyValidator: penalty={penalty:.3f} "
            f"art_profile={art_profile} mod_profile={mod_profile}"
        )
        return penalty, debug

    def apply_penalty(
        self,
        resonance: float,
        art_four_d: Dict,
        model: Dict,
    ) -> Tuple[float, float, Dict]:
        """
        Применяет топологический штраф к резонансу.

        Returns:
            (adjusted_resonance, penalty, debug_info)
        """
        penalty, debug = self.compute_penalty(art_four_d, model)
        adjusted = round(resonance * penalty, 4)
        debug["original_resonance"]  = resonance
        debug["adjusted_resonance"]  = adjusted
        return adjusted, penalty, debug
