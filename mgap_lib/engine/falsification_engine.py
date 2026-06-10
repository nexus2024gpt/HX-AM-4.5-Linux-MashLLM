# mgap_lib/engine/falsification_engine.py — HX-AM v4.7
"""
Контур фальсификации (Критерий Поппера для MGAP).

Проблема:
  _build_verdict() ищет аргументы «за» (why_applicable).
  Научная верификация требует поиска условий, при которых
  модель ГАРАНТИРОВАННО не работает.

Решение:
  FalsificationEngine содержит математически строгие правила
  фальсификации для каждого math_type. Правила работают
  детерминированно — без LLM, на основе числовых параметров.

  Если хотя бы одно правило сработало (triggered=True) →
  resonance умножается на INVALIDATION_PENALTY.

Правила фальсификации (дополнены):

  Общие для всех типов:
    - tau_non_positive: τ ≤ 0 → нарушение причинности
    - eta_negative: η < 0 → нефизично
    - nan_inf: любые параметры NaN/Inf → INVALIDATED

  percolation:
    - giant_component_collapse: p → 1.0 → вырождение
    - molloy_reed_violation: Scale-Free граф, но порог ER
    - p_crit_too_low: p_crit < 0.01 → модель нечувствительна

  kuramoto:
    - mean_field_breakdown: k < 5 → разреженная сеть
    - frequency_spread_collapse: ω/K > 5 → синхронизация невозможна
    - K_zero: K ≤ 0 → нет связи
    - omega_i_negative: ω < 0 → нефизично

  delay:
    - pade_singularity: K·τ > π/2 → гарантированная неустойчивость
    - tau_negative: τ < 0 → нарушение причинности

  ising:
    - mean_field_low_dimension: D < 2 → MF неприменима
    - critical_fluctuations_divergence: |T-K|/K < 0.05 → флуктуации
    - temperature_out_of_range: T < 0 or T > 10 → вне диапазона

  lotka_volterra:
    - negative_equilibrium: K≤0 or p≤0 or K_c≤0 or ω≤0
    - oscillation_collapse: ω·K_c < 0.01 → апериодичность
    - too_small_equilibrium: x*<0.01 or y*<0.01 → близость к вымиранию

  graph_invariant:
    - below_percolation_threshold: p < 1/k → граф несвязен
    - p_critical_too_low: p_crit < 0.01
    - k_zero: k < 1 → деление на ноль
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional

logger = logging.getLogger("HXAM.mgap.falsification")

# Коэффициент снижения резонанса при срабатывании фальсификации
INVALIDATION_PENALTY = 0.5

# Пороги
_PADE_THRESHOLD = math.pi / 2               # ≈ 1.5708
_MIN_K_MEAN_FIELD = 5.0
_FREQ_SPREAD_RATIO_MAX = 5.0
_P_SINGULARITY_UPPER = 0.995
_LV_MIN_EQUILIBRIUM = 1e-3
_MIN_P_CRIT = 0.01
_TEMP_MIN = 0.0
_TEMP_MAX = 10.0


class FalsificationRule:
    """Описание одного правила фальсификации."""
    def __init__(
        self,
        name: str,
        description: str,
        boundary: str,
        severity: str = "INVALIDATED",
    ):
        self.name = name
        self.description = description
        self.boundary = boundary
        self.severity = severity


# Реестр правил по math_type (включая общие)
_COMMON_RULES = [
    FalsificationRule(
        name="tau_non_positive",
        description="τ ≤ 0: задержка не может быть нулевой или отрицательной (нарушение причинности).",
        boundary="tau > 0",
        severity="INVALIDATED",
    ),
    FalsificationRule(
        name="eta_negative",
        description="η < 0: уровень шума не может быть отрицательным.",
        boundary="eta >= 0",
        severity="INVALIDATED",
    ),
    FalsificationRule(
        name="nan_inf",
        description="Обнаружены значения NaN или Inf в параметрах модели.",
        boundary="Все параметры конечны",
        severity="INVALIDATED",
    ),
]

_RULES: Dict[str, List[FalsificationRule]] = {

    "percolation": _COMMON_RULES + [
        FalsificationRule(
            name="giant_component_collapse",
            description="При p → 1.0 модель перколяции вырождается: все узлы связаны, понятие «гигантской компоненты» теряет смысл.",
            boundary=f"p < {_P_SINGULARITY_UPPER}",
        ),
        FalsificationRule(
            name="molloy_reed_violation",
            description="Артефакт имеет характеристики Scale-Free сети, но модель использует порог ER-графа p_crit=1/k. Для безмасштабных сетей корректный порог Molloy-Reed: p_crit = <k> / (<k²> - <k>).",
            boundary="Если топология Scale-Free: используй критерий Molloy-Reed",
            severity="WARNING",
        ),
        FalsificationRule(
            name="p_crit_too_low",
            description=f"Порог перколяции p_crit < {_MIN_P_CRIT} — модель нечувствительна и вырождается.",
            boundary=f"p_crit >= {_MIN_P_CRIT}",
            severity="WARNING",
        ),
    ],

    "kuramoto": _COMMON_RULES + [
        FalsificationRule(
            name="mean_field_breakdown",
            description=f"Модель Курамото предполагает all-to-all связи (mean-field приближение). При средней степени k < {_MIN_K_MEAN_FIELD} сеть разрежена — фазовый переход размывается.",
            boundary=f"k >= {_MIN_K_MEAN_FIELD}",
        ),
        FalsificationRule(
            name="frequency_spread_collapse",
            description=f"Если дисперсия собственных частот omega_i значительно больше силы связи K, синхронизация невозможна. Критическое отношение: omega_i/K < {_FREQ_SPREAD_RATIO_MAX}.",
            boundary=f"omega_i / K < {_FREQ_SPREAD_RATIO_MAX}",
        ),
        FalsificationRule(
            name="K_zero",
            description="K ≤ 0: сила связи не может быть нулевой или отрицательной.",
            boundary="K > 0",
            severity="INVALIDATED",
        ),
        FalsificationRule(
            name="omega_i_negative",
            description="ω < 0: собственная частота не может быть отрицательной.",
            boundary="ω >= 0",
            severity="INVALIDATED",
        ),
    ],

    "delay": _COMMON_RULES + [
        FalsificationRule(
            name="pade_singularity",
            description=f"Критерий устойчивости Падé для систем с запаздыванием: при K·τ > π/2 ≈ {_PADE_THRESHOLD:.3f} система гарантированно неустойчива.",
            boundary=f"K * tau < {_PADE_THRESHOLD:.3f}",
        ),
        FalsificationRule(
            name="tau_negative",
            description="τ < 0: задержка отрицательна – нарушение причинности.",
            boundary="τ >= 0",
            severity="INVALIDATED",
        ),
    ],

    "ising": _COMMON_RULES + [
        FalsificationRule(
            name="mean_field_low_dimension",
            description="Mean-field Ising-модель неприменима при пространственной размерности D < 2 (например, цепочка Ising не имеет фазового перехода).",
            boundary="D >= 2.0",
        ),
        FalsificationRule(
            name="critical_fluctuations_divergence",
            description="Вблизи T_c критические флуктуации расходятся. При |T - K|/K < 0.05 предсказания MF количественно неверны.",
            boundary="|T - K|/K > 0.05",
            severity="WARNING",
        ),
        FalsificationRule(
            name="temperature_out_of_range",
            description=f"Температура T вне физического диапазона: T < {_TEMP_MIN} или T > {_TEMP_MAX}.",
            boundary=f"{_TEMP_MIN} <= T <= {_TEMP_MAX}",
            severity="WARNING",
        ),
    ],

    "lotka_volterra": _COMMON_RULES + [
        FalsificationRule(
            name="negative_equilibrium",
            description="Стационарное состояние требует K>0, p>0, K_c>0, ω>0. Иначе x* = K_c/p или y* = ω/K становятся неположительными, теряя физический смысл.",
            boundary="K > 0, p > 0, K_c > 0, ω > 0",
            severity="INVALIDATED",
        ),
        FalsificationRule(
            name="oscillation_collapse",
            description="Период колебаний T ≈ 2π/√(ω·K_c). При ω·K_c < 0.01 период → ∞, модель теряет предсказательную силу для циклических систем.",
            boundary="ω * K_c >= 0.01",
            severity="WARNING",
        ),
        FalsificationRule(
            name="too_small_equilibrium",
            description="Равновесные концентрации x* = K_c/p или y* = ω/K близки к нулю (<0.01) – система на грани вымирания.",
            boundary="x* >= 0.01 и y* >= 0.01",
            severity="WARNING",
        ),
    ],

    "graph_invariant": _COMMON_RULES + [
        FalsificationRule(
            name="below_percolation_threshold",
            description="При p < p_c = 1/k граф несвязен: нет гигантской компоненты. Глобальные граф-инварианты (betweenness, community structure) теряют смысл.",
            boundary="p > 1/k",
        ),
        FalsificationRule(
            name="p_critical_too_low",
            description=f"Порог связности p_crit = 1/k < {_MIN_P_CRIT} — граф практически всегда связен, модель вырождается.",
            boundary=f"p_crit >= {_MIN_P_CRIT}",
            severity="WARNING",
        ),
        FalsificationRule(
            name="k_zero",
            description="k < 1: средняя степень графа меньше 1, граф не может быть связным; деление на k невозможно.",
            boundary="k >= 1",
            severity="INVALIDATED",
        ),
    ],
}

# Добавим для "delay_ode", "delay-ode" те же правила, что и для "delay"
_RULES["delay_ode"] = _RULES["delay"]
_RULES["delay-ode"] = _RULES["delay"]


# ----- Функции проверки для каждого math_type (включая общие) -----
def _check_common(flat: Dict) -> Dict[str, bool]:
    """Общие проверки: tau, eta, NaN/Inf."""
    triggered = {}
    # NaN/Inf
    for key in ["tau", "eta", "K", "p", "omega_i", "k", "C", "D", "T"]:
        val = flat.get(key)
        if val is not None and (math.isnan(val) or math.isinf(val)):
            triggered[f"{key}_nan_inf"] = True
            # Для упрощения – одно правило nan_inf уже есть
    # tau_non_positive
    tau = flat.get("tau")
    if tau is not None and tau <= 0:
        triggered["tau_non_positive"] = True
    # eta_negative
    eta = flat.get("eta")
    if eta is not None and eta < 0:
        triggered["eta_negative"] = True
    return triggered


def _check_percolation(flat: Dict, model: Dict) -> Dict[str, bool]:
    p = float(flat.get("p", 0.65))
    C = float(flat.get("C", 0.5))
    D = float(flat.get("D", 2.0))

    triggered = _check_common(flat)

    # giant_component_collapse
    if p > _P_SINGULARITY_UPPER:
        triggered["giant_component_collapse"] = True

    # molloy_reed_violation
    is_scale_free_like = (D > 2.4 and C > 0.58)
    ct = model.get("critical_thresholds", {})
    has_er_p_crit = ("p_crit" in ct and not ct.get("_dynamic"))
    if is_scale_free_like and has_er_p_crit:
        triggered["molloy_reed_violation"] = True

    # p_crit_too_low
    p_crit = float(ct.get("p_crit", 0.5))
    if p_crit < _MIN_P_CRIT:
        triggered["p_crit_too_low"] = True

    return triggered


def _check_kuramoto(flat: Dict) -> Dict[str, bool]:
    k = float(flat.get("k", 6.0))
    K = float(flat.get("K", 0.35))
    omega_i = float(flat.get("omega_i", 0.25))

    triggered = _check_common(flat)

    if k < _MIN_K_MEAN_FIELD:
        triggered["mean_field_breakdown"] = True

    ratio = omega_i / max(K, 1e-9)
    if ratio > _FREQ_SPREAD_RATIO_MAX:
        triggered["frequency_spread_collapse"] = True

    if K <= 0:
        triggered["K_zero"] = True
    if omega_i < 0:
        triggered["omega_i_negative"] = True

    return triggered


def _check_delay(flat: Dict) -> Dict[str, bool]:
    K = float(flat.get("K", 0.35))
    tau = float(flat.get("tau", 0.5))

    triggered = _check_common(flat)

    if K * tau > _PADE_THRESHOLD:
        triggered["pade_singularity"] = True
    if tau < 0:
        triggered["tau_negative"] = True

    return triggered


def _check_ising(flat: Dict) -> Dict[str, bool]:
    D = float(flat.get("D", 2.0))
    T = float(flat.get("T", 1.0))
    K = float(flat.get("K", 0.35))

    triggered = _check_common(flat)

    if D < 2.0:
        triggered["mean_field_low_dimension"] = True

    T_c = K
    if T_c > 1e-9:
        relative_dist = abs(T - T_c) / T_c
        if relative_dist < 0.05:
            triggered["critical_fluctuations_divergence"] = True
    # Иначе T_c=0 — не проверяем

    if T < _TEMP_MIN or T > _TEMP_MAX:
        triggered["temperature_out_of_range"] = True

    return triggered


def _check_lotka_volterra(flat: Dict) -> Dict[str, bool]:
    K = float(flat.get("K", 0.35))
    K_c = float(flat.get("K_c", 0.48))
    p = float(flat.get("p", 0.65))
    omega_i = float(flat.get("omega_i", 0.25))

    triggered = _check_common(flat)

    if K <= 0 or p <= 0 or K_c <= 0 or omega_i <= 0:
        triggered["negative_equilibrium"] = True

    if omega_i * K_c < 0.01:
        triggered["oscillation_collapse"] = True

    x_star = K_c / max(p, 1e-9)
    y_star = omega_i / max(K, 1e-9)
    if x_star < 0.01 or y_star < 0.01:
        triggered["too_small_equilibrium"] = True

    return triggered


def _check_graph_invariant(flat: Dict) -> Dict[str, bool]:
    p = float(flat.get("p", 0.65))
    k = float(flat.get("k", 6.0))

    triggered = _check_common(flat)

    p_c = 1.0 / max(k, 1e-9)
    if p < p_c:
        triggered["below_percolation_threshold"] = True

    if p_c < _MIN_P_CRIT:
        triggered["p_critical_too_low"] = True

    if k < 1.0:
        triggered["k_zero"] = True

    return triggered


_CHECKERS = {
    "percolation": _check_percolation,
    "kuramoto": _check_kuramoto,
    "delay": _check_delay,
    "ising": _check_ising,
    "lotka_volterra": _check_lotka_volterra,
    "graph_invariant": _check_graph_invariant,
}


class FalsificationEngine:
    """
    Генерирует сценарии разрушения для каждого матча.

    Пример:
        fe = FalsificationEngine()
        flat = {"K": 1.8, "tau": 1.5, "k": 6.0, ...}
        model = {"math_type": "delay"}
        result = fe.run(flat, model)
        # result["overall_verdict"] == "INVALIDATED"
        # result["resonance_multiplier"] == 0.5
    """

    def run(
        self,
        flat: Dict,
        model: Dict,
    ) -> Dict[str, Any]:
        """
        Запускает все правила фальсификации для данного math_type.

        flat: параметры артефакта из _flat_4d() + structure поля
        model: модель из реестра

        Returns:
            {
                "scenarios": List[Dict],
                "invalidated_count": int,
                "warning_count": int,
                "overall_verdict": "OK" | "WARNING" | "INVALIDATED",
                "critical_warnings": List[str],
                "warning_descriptions": List[str],
                "resonance_multiplier": float,
            }
        """
        mt = str(model.get("math_type", "")).lower().strip()

        if mt in ("delay_ode", "delay-ode"):
            mt = "delay"

        rules = _RULES.get(mt, [])
        checker = _CHECKERS.get(mt)

        if not rules or checker is None:
            return self._empty_result()

        # Вызов проверщика (разная сигнатура для percolation)
        if mt == "percolation":
            triggered_map = checker(flat, model)
        else:
            triggered_map = checker(flat)

        scenarios = []
        invalidated = []
        warnings_list = []

        for rule in rules:
            is_triggered = triggered_map.get(rule.name, False)
            scenarios.append({
                "name": rule.name,
                "triggered": is_triggered,
                "severity": rule.severity if is_triggered else "OK",
                "condition": rule.boundary,
                "description": rule.description,
            })
            if is_triggered:
                if rule.severity == "INVALIDATED":
                    invalidated.append(rule)
                else:
                    warnings_list.append(rule)

        if invalidated:
            overall = "INVALIDATED"
            multiplier = INVALIDATION_PENALTY
        elif warnings_list:
            overall = "WARNING"
            multiplier = 1.0 - 0.1 * len(warnings_list)  # -10% за каждое предупреждение
            multiplier = max(0.6, round(multiplier, 2))
        else:
            overall = "OK"
            multiplier = 1.0

        critical = [r.description for r in invalidated]
        warning_descs = [r.description for r in warnings_list]

        result = {
            "scenarios": scenarios,
            "invalidated_count": len(invalidated),
            "warning_count": len(warnings_list),
            "overall_verdict": overall,
            "critical_warnings": critical,
            "warning_descriptions": warning_descs,
            "resonance_multiplier": multiplier,
        }

        if invalidated or warnings_list:
            logger.info(
                f"FalsificationEngine [{mt}]: verdict={overall} "
                f"invalidated={len(invalidated)} warnings={len(warnings_list)}"
            )

        return result

    @staticmethod
    def _empty_result() -> Dict[str, Any]:
        return {
            "scenarios": [],
            "invalidated_count": 0,
            "warning_count": 0,
            "overall_verdict": "OK",
            "critical_warnings": [],
            "warning_descriptions": [],
            "resonance_multiplier": 1.0,
        }
