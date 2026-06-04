# mgap_matcher.py — HX-AM v4.6
"""
MGAPMatcher — переносит численные инварианты из артефактов HX-AM
в прикладные отраслевые модели (реестр MGAP).

v4.5 исправления:
  - Добавлена кодогенерация для math_type=ising
  - Добавлен расчёт на примере для type=ising
  - Verdict учитывает survival_verified (False → предупреждение)
  - Verdict учитывает stability_score < 0.5 (математически нестабильный)
  - _compute_resonance: нормализован type_bonus к правильным весам
  - v4.5.6: retry для blind_spot и match_analysis

v4.6 исправления:
  - _calculate_example использует p_crit из critical_thresholds модели
  - graph_invariant поддерживает отраслевые контексты по logia
  - percolation example синхронизирован с модельными порогами
    - Применены правки из `mgap_matcher_ref.txt` (2026-05-31):
        расширены отраслевые контексты, улучшена генерация слепых зон
        и форматирования шаблонов с использованием `critical_thresholds`.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("HXAM.mgap")

_MATH_TYPE_ALIASES: Dict[str, str] = {
    "delay_ode": "delay",
    "delay-ode": "delay",
    "graph-invariant": "graph_invariant",
}

ARTIFACTS_DIR = Path("artifacts")
REGISTRY_PATH = Path("mgap_registry.json")

LOGIA_GRAPH_INVARIANT_CONTEXT: Dict[str, Dict[str, Any]] = {
    "Логистика": {
        "flow_name":    "daily_sales",
        "flow_label":   "Среднедневные продажи",
        "lag_label":    "Срок поставки (дней)",
        "buffer_label": "Страховой запас (коэф.)",
        "flow_mean":  100.0,  "flow_std":  30.0,  "lag": 3.0,  "old_coef": 0.2,
    },
    "Астрономия": {
        "flow_name":    "orbital_velocity",
        "flow_label":   "Орбитальная скорость (усл. ед.)",
        "lag_label":    "Время передачи сигнала (с)",
        "buffer_label": "Запас стабильности орбиты (коэф.)",
        "flow_mean":  1.0,    "flow_std":  0.3,   "lag": 2.0,  "old_coef": 0.15,
    },
    "Экология": {
        "flow_name":    "biomass_flow",
        "flow_label":   "Поток биомассы (т/год)",
        "lag_label":    "Время оборота популяции (лет)",
        "buffer_label": "Резервный запас биомассы (коэф.)",
        "flow_mean":  500.0,  "flow_std": 150.0,  "lag": 5.0,  "old_coef": 0.25,
    },
    "Инженерия": {
        "flow_name":    "throughput",
        "flow_label":   "Пропускная способность (ед./с)",
        "lag_label":    "Задержка передачи (мс)",
        "buffer_label": "Резерв буфера (коэф.)",
        "flow_mean":  1000.0, "flow_std": 200.0,  "lag": 1.5,  "old_coef": 0.1,
    },
    "Политология": {
        "flow_name":    "influence_flow",
        "flow_label":   "Поток влияния (индекс)",
        "lag_label":    "Задержка реакции системы (мес.)",
        "buffer_label": "Резерв институциональной устойчивости",
        "flow_mean":  50.0,   "flow_std":  20.0,  "lag": 6.0,  "old_coef": 0.3,
    },
    "Социология": {
        "flow_name":    "information_flow",
        "flow_label":   "Поток информации (ед./час)",
        "lag_label":    "Время распространения (часы)",
        "buffer_label": "Буфер фильтрации контента",
        "flow_mean":  200.0,  "flow_std":  80.0,  "lag": 4.0,  "old_coef": 0.2,
    },
    "Технологии": {
        "flow_name":    "request_rate",
        "flow_label":   "Частота запросов (req/s)",
        "lag_label":    "Задержка ответа (мс)",
        "buffer_label": "Буфер очереди (коэф.)",
        "flow_mean":  500.0,  "flow_std": 100.0,  "lag": 0.8,  "old_coef": 0.15,
    },
    "Геонауки": {
        "flow_name":    "seismic_energy",
        "flow_label":   "Сейсмическая энергия (усл. ед.)",
        "lag_label":    "Время накопления напряжений (лет)",
        "buffer_label": "Запас сейсмической прочности (коэф.)",
        "flow_mean":  300.0,  "flow_std": 120.0,  "lag": 8.0,  "old_coef": 0.35,
    },
    "Океанография": {
        "flow_name":    "current_velocity",
        "flow_label":   "Скорость течения (м/с)",
        "lag_label":    "Время перестройки циркуляции (лет)",
        "buffer_label": "Запас устойчивости циркуляции",
        "flow_mean":  0.5,    "flow_std":  0.2,   "lag": 10.0, "old_coef": 0.2,
    },
    "Физика": {
        "flow_name":    "energy_flux",
        "flow_label":   "Поток энергии (Вт/м²)",
        "lag_label":    "Время релаксации (с)",
        "buffer_label": "Буфер энергетического резерва",
        "flow_mean":  100.0,  "flow_std":  25.0,  "lag": 1.0,  "old_coef": 0.15,
    },
    "Материаловедение": {
        "flow_name":    "strain_rate",
        "flow_label":   "Скорость деформации (1/с)",
        "lag_label":    "Время релаксации напряжений (с)",
        "buffer_label": "Запас прочности материала",
        "flow_mean":  0.01,   "flow_std":  0.003, "lag": 0.5,  "old_coef": 0.2,
    },
    "Химия": {
        "flow_name":    "reaction_rate",
        "flow_label":   "Скорость реакции (моль/л·с)",
        "lag_label":    "Индукционный период (с)",
        "buffer_label": "Запас реагентов (коэф.)",
        "flow_mean":  0.1,    "flow_std":  0.04,  "lag": 2.0,  "old_coef": 0.3,
    },
    "Биология": {
        "flow_name":    "population_density",
        "flow_label":   "Плотность популяции (ос./км²)",
        "lag_label":    "Время генерации (лет)",
        "buffer_label": "Резервная ёмкость среды (коэф.)",
        "flow_mean":  1000.0, "flow_std": 300.0,  "lag": 3.0,  "old_coef": 0.25,
    },
    "Экономика": {
        "flow_name":    "transaction_volume",
        "flow_label":   "Объём транзакций (ед./день)",
        "lag_label":    "Лаг реакции рынка (дней)",
        "buffer_label": "Резервный капитал (коэф.)",
        "flow_mean":  10000.0,"flow_std":3000.0,  "lag": 5.0,  "old_coef": 0.15,
    },
    "Междисциплинарно": {
        "flow_name":    "system_flux",
        "flow_label":   "Системный поток (усл. ед.)",
        "lag_label":    "Задержка обратной связи (усл. ед.)",
        "buffer_label": "Буфер системной устойчивости (коэф.)",
        "flow_mean":  100.0,  "flow_std":  30.0,  "lag": 3.0,  "old_coef": 0.2,
    },
}

_DEFAULT_GI_CONTEXT: Dict[str, Any] = {
    "flow_name":    "flow_value",
    "flow_label":   "Значение потока",
    "lag_label":    "Задержка (усл. ед.)",
    "buffer_label": "Коэффициент буфера",
    "flow_mean":  100.0, "flow_std": 30.0, "lag": 3.0, "old_coef": 0.2,
}


def _norm_math_type(t: str) -> str:
    return _MATH_TYPE_ALIASES.get(t.lower().strip(), t.lower().strip())


def _format_blind_spot(
    template: str,
    eta_crit: float,
    tau_crit: float,
    p_crit: float = 0.37,
    K_min: float = 0.0,
    T_crit: float = 0.0,
    p: float = 0.5,
) -> str:
    """
    Форматирует blind_spot_template, подставляя актуальные пороги модели.
    Поддерживает оба формата плейсхолдеров: {eta_max}/{tau_max} (старый)
    и {eta_crit}/{tau_crit} (новый). Жёсткие числа в шаблоне остаются
    как есть — функция только подставляет плейсхолдеры.
    """
    if not template:
        return template
    try:
        return template.format(
            # Новые плейсхолдеры
            eta_crit=round(eta_crit, 3),
            tau_crit=round(tau_crit, 3),
            p_crit=round(p_crit, 3),
            K_min=round(K_min, 3),
            T_crit=round(T_crit, 3),
            # Обратная совместимость со старыми плейсхолдерами
            eta_max=round(eta_crit, 3),
            tau_max=round(tau_crit, 3),
            p=round(p, 3),
        )
    except (KeyError, IndexError):
        # Если в шаблоне неизвестный плейсхолдер — вернуть как есть
        return template



def _extract_art_four_d(artifact: Dict) -> Optional[Dict]:
    return artifact.get("data", {}).get("gen", {}).get("four_d_matrix")


def _extract_art_sim(artifact: Dict) -> Dict:
    return artifact.get("simulation") or {}


def _extract_thresholds(sim: Dict, ver: Dict, model: Dict) -> Dict:
    stress = ver.get("stress_test") or {}
    ct     = model.get("critical_thresholds", {})

    eta = stress.get("eta_critical") or ct.get("eta_max") or \
          sim.get("eta_critical") or sim.get("bifurcation_boundary", {}).get("eta_max", 0.5)
    tau = stress.get("tau_robustness") or ct.get("tau_max") or \
          sim.get("tau_robustness") or sim.get("bifurcation_boundary", {}).get("tau_max_stable", 1.0)

    return {
        "eta_critical":     float(eta),
        "tau_robustness":   float(tau),
        "lyapunov_max":     float(sim.get("lyapunov_max", 0.0)),
        "stability_score":  float(sim.get("stability_score", 0.5)),
        "survival_verified": bool(sim.get("survival_verified", False)),
    }


def _flat_4d(four_d: Dict) -> Dict[str, Any]:
    dyn = four_d.get("dynamics", {})
    inf = four_d.get("influence", {})
    tim = four_d.get("time", {})
    return {
        "tau":     float(tim.get("tau", 0.5)),
        "K":       float(dyn.get("K", 0.35)),
        "K_c":     float(dyn.get("K_c", 0.48)),
        "eta":     float(inf.get("eta", 0.2)),
        "omega_i": float(dyn.get("omega_i", 0.25)),
        "p":       float(dyn.get("p", 0.65)),
        "model":   str(dyn.get("model", "kuramoto")),
    }


# ══════════════════════════════════════════════════════════
# РЕЗОНАНС
# ══════════════════════════════════════════════════════════

def _art_vector(four_d: Dict) -> Optional[np.ndarray]:
    try:
        from schemas.four_d_matrix import FourDMatrix
        m = FourDMatrix.from_raw(four_d)
        return m.to_vector() if m else None
    except Exception:
        return None


def _model_vector(model: Dict) -> Optional[np.ndarray]:
    return _art_vector(model.get("four_d_matrix", {}))


def _compute_resonance(art_vec: Optional[np.ndarray], model: Dict, art_math: str) -> float:
    if art_vec is None:
        return _resonance_fallback(_flat_4d({}), model)
    try:
        from schemas.four_d_matrix import compute_4d_resonance
        m_vec = _model_vector(model)
        if m_vec is None:
            return 0.0
        vec_res = float(compute_4d_resonance(art_vec, m_vec))
    except Exception:
        vec_res = 0.0

    type_bonus = 0.3 if _norm_math_type(model.get("math_type", "")) == _norm_math_type(art_math) else 0.0
    return round(vec_res * 0.7 + type_bonus, 3)


def _resonance_fallback(flat: Dict, model: Dict) -> float:
    m4d    = model.get("four_d_matrix") or {}
    m_flat = _flat_4d(m4d)
    ranges  = model.get("expected_ranges") or {}
    weights = model.get("weights") or {}
    total = score = 0.0
    for key in ("tau", "K", "eta"):
        r   = ranges.get(key, [0.0, 1.0])
        lo, hi = (r[0], r[1]) if isinstance(r, list) and len(r) == 2 else (0.0, 1.0)
        w    = float(weights.get(key, 1.0))
        span = max(hi - lo, 1e-9)
        sim  = max(0.0, 1.0 - abs(flat.get(key, 0.5) - m_flat.get(key, 0.5)) / span)
        score += sim * w
        total += w
    return round(score / max(total, 1e-9), 3)


# ══════════════════════════════════════════════════════════
# КОДОГЕНЕРАЦИЯ — все 6 math_type
# ══════════════════════════════════════════════════════════

def _generate_code(model: Dict, thresholds: Dict, flat: Dict) -> str:
    mt    = _norm_math_type(model.get("math_type", "kuramoto"))
    eta_c = thresholds["eta_critical"]
    tau_c = thresholds["tau_robustness"]
    prog  = (model.get("programs") or ["target_system"])[0]

    if mt == "graph_invariant":
        return (
            f"# MGAP Stability Monitor — {prog}\n"
            f"# model=graph_invariant, eta_crit={eta_c:.3f}, tau_crit={tau_c:.3f}\n"
            f"def mgap_stability_monitor(flow_values, lag_values, old_buffer_coef=0.2):\n"
            f"    import numpy as np\n"
            f"    eta = np.std(flow_values) / max(np.mean(flow_values), 1e-9)\n"
            f"    tau = np.mean(lag_values)\n"
            f"    warn = (eta > {eta_c:.3f}) or (tau > {tau_c:.3f})\n"
            f"    if warn:\n"
            f"        mult    = max(1.0, (eta / {eta_c:.3f}) * (tau / {tau_c:.3f}))\n"
            f"        new_buf = old_buffer_coef * mult\n"
            f"        return {{'warning': True, 'multiplier': round(mult, 3),\n"
            f"                'new_buffer_coef': round(new_buf, 3)}}\n"
            f"    return {{'warning': False, 'multiplier': 1.0, 'new_buffer_coef': old_buffer_coef}}\n"
        )

    elif mt == "kuramoto":
        K_c = model.get("critical_thresholds", {}).get("K_min", flat.get("K_c", 0.5))
        return (
            f"# MGAP Stability Monitor — {prog}\n"
            f"# model=kuramoto, K_c={K_c:.3f}, eta_crit={eta_c:.3f}, tau_crit={tau_c:.3f}\n"
            f"def mgap_stability_monitor(coupling_K, noise_eta, delay_tau):\n"
            f"    warnings = []\n"
            f"    if coupling_K < {K_c:.3f}:\n"
            f"        warnings.append(f'K={{coupling_K:.3f}} < K_c={K_c:.3f}')\n"
            f"    if noise_eta > {eta_c:.3f}:\n"
            f"        warnings.append(f'η={{noise_eta:.3f}} > η_crit={eta_c:.3f}')\n"
            f"    if delay_tau > {tau_c:.3f}:\n"
            f"        warnings.append(f'τ={{delay_tau:.3f}} > τ_crit={tau_c:.3f}')\n"
            f"    stable = len(warnings) == 0\n"
            f"    return {{'stable': stable, 'warnings': warnings}}\n"
        )

    elif mt in ("delay", "delay_ode"):
        K_min = model.get("critical_thresholds", {}).get("K_min", 0.1)
        return (
            f"# MGAP Stability Monitor — {prog}\n"
            f"# model=delay, eta_crit={eta_c:.3f}, tau_crit={tau_c:.3f}, K_min={K_min:.3f}\n"
            f"def mgap_stability_margin(eta, tau, K):\n"
            f"    margin = min(\n"
            f"        1 - eta / {eta_c:.3f},\n"
            f"        1 - tau / {tau_c:.3f},\n"
            f"        (K - {K_min:.3f}) / max({K_min:.3f}, 1e-9),\n"
            f"    )\n"
            f"    return {{'stability_margin': round(margin, 4), 'warning': margin < 0.2}}\n"
        )

    elif mt == "ising":
        K_min = model.get("critical_thresholds", {}).get("K_min", 0.4)
        T_crit = model.get("critical_thresholds", {}).get("T_crit", 1.0)
        return (
            f"# MGAP Stability Monitor — {prog}\n"
            f"# model=ising  T_crit={T_crit:.3f}, eta_crit={eta_c:.3f}, tau_crit={tau_c:.3f}\n"
            f"# Упорядоченная фаза: T < T_crit (намагниченность / консенсус норм)\n"
            f"import math\n"
            f"\n"
            f"def mgap_ising_phase_check(T_temperature, eta_fluctuation, K_coupling, tau_relax):\n"
            f"    \"\"\"\n"
            f"    T_temperature  : фактическая температура / стохастичность системы\n"
            f"    eta_fluctuation: уровень случайных флуктуаций (0–1)\n"
            f"    K_coupling     : сила взаимодействия между элементами (нормирована)\n"
            f"    tau_relax      : время релаксации системы к равновесию\n"
            f"    \"\"\"\n"
            f"    # Упорядоченная фаза требует T < T_crit = K\n"
            f"    T_c = K_coupling  # T_crit ≈ K в mean-field Ising\n"
            f"    warnings = []\n"
            f"    if T_temperature >= T_c:\n"
            f"        warnings.append(\n"
            f"            f'T={{T_temperature:.3f}} >= T_c={{T_c:.3f}}: система в неупорядоченной фазе'\n"
            f"        )\n"
            f"    if eta_fluctuation > {eta_c:.3f}:\n"
            f"        warnings.append(\n"
            f"            f'η={{eta_fluctuation:.3f}} > η_crit={eta_c:.3f}: флуктуации разрушают порядок'\n"
            f"        )\n"
            f"    if tau_relax > {tau_c:.3f}:\n"
            f"        warnings.append(\n"
            f"            f'τ={{tau_relax:.3f}} > τ_crit={tau_c:.3f}: релаксация слишком медленная'\n"
            f"        )\n"
            f"    try:\n"
            f"        m = math.tanh(K_coupling / max(T_temperature, 0.01))\n"
            f"    except Exception:\n"
            f"        m = 0.0\n"
            f"    order_param = round(abs(m) * (1 - eta_fluctuation), 3)\n"
            f"    stable = len(warnings) == 0 and order_param > 0.3\n"
            f"    return {{\n"
            f"        'stable':       stable,\n"
            f"        'order_param':  order_param,\n"
            f"        'T_c_approx':   round(T_c, 3),\n"
            f"        'warnings':     warnings,\n"
            f"        'recommendation': 'Система в упорядоченной фазе.' if stable else\n"
            f"                          f'Снизить T ниже {{round(T_c, 3)}} и η ниже {eta_c:.3f}.'\n"
            f"    }}\n"
        )

    elif mt == "percolation":
        p_crit = model.get("critical_thresholds", {}).get("p_crit", 0.37)
        return (
            f"# MGAP Stability Monitor — {prog}\n"
            f"# model=percolation, p_crit={p_crit:.3f}, eta_crit={eta_c:.3f}\n"
            f"def mgap_percolation_risk(p_connectivity, eta_heterogeneity):\n"
            f"    above_threshold = p_connectivity > {p_crit:.3f}\n"
            f"    cascade_risk = max(0.0, (p_connectivity - {p_crit:.3f}) / (1 - {p_crit:.3f}))\n"
            f"    eta_penalty  = eta_heterogeneity / max({eta_c:.3f}, 1e-9)\n"
            f"    compound_risk = round(cascade_risk * max(1.0, eta_penalty), 3)\n"
            f"    return {{\n"
            f"        'above_threshold': above_threshold,\n"
            f"        'cascade_risk':    round(cascade_risk, 3),\n"
            f"        'compound_risk':   compound_risk,\n"
            f"        'warning':         compound_risk > 0.3,\n"
            f"        'recommendation':  'Снизить связность ниже p_crit={p_crit:.3f}.' if above_threshold else 'Система ниже порога.'\n"
            f"    }}\n"
        )

    elif mt == "lotka_volterra":
        K_min = model.get("critical_thresholds", {}).get("K_min", 0.2)
        return (
            f"# MGAP Stability Monitor — {prog}\n"
            f"# model=lotka_volterra, eta_crit={eta_c:.3f}, tau_crit={tau_c:.3f}\n"
            f"def mgap_lv_coexistence_check(K_interaction, eta_resource_variance, tau_cycle):\n"
            f"    warnings = []\n"
            f"    if K_interaction > {K_min + 0.3:.3f}:  # выше критической конкуренции\n"
            f"        warnings.append(f'K={{K_interaction:.3f}} > порога: конкуренция подавляет коэксистенцию')\n"
            f"    if eta_resource_variance > {eta_c:.3f}:\n"
            f"        warnings.append(f'η={{eta_resource_variance:.3f}} > η_crit={eta_c:.3f}: ресурсная нестабильность')\n"
            f"    if tau_cycle > {tau_c:.3f}:\n"
            f"        warnings.append(f'τ={{tau_cycle:.3f}} > τ_crit={tau_c:.3f}: цикл слишком длинный')\n"
            f"    coexistence_stable = len(warnings) == 0\n"
            f"    return {{'coexistence_stable': coexistence_stable, 'warnings': warnings}}\n"
        )

    return f"# math_type '{mt}' — code snippet not yet implemented in MGAP v4.5"


# ══════════════════════════════════════════════════════════
# РАСЧЁТ НА ПРИМЕРЕ — все 6 math_type
# ══════════════════════════════════════════════════════════

def _calculate_example(model: Dict, thresholds: Dict) -> Dict:
    result = _calculate_example_raw(model, thresholds)
    result["is_synthetic"] = True
    result["_note"] = "Синтетический пример из реестра. Не используется для ROI и вердикта."
    return result


def _calculate_example_raw(model: Dict, thresholds: Dict) -> Dict:
    example = model.get("example_data") or {}
    eta_c   = thresholds["eta_critical"]
    tau_c   = thresholds["tau_robustness"]
    t       = example.get("type", "graph_invariant")
    logia   = model.get("logia", "")
    ct      = model.get("critical_thresholds", {})

    if t == "graph_invariant":
        ctx = LOGIA_GRAPH_INVARIANT_CONTEXT.get(logia, _DEFAULT_GI_CONTEXT)

        d_mean  = float(example.get("daily_sales_mean",
                        example.get("flow_mean", ctx["flow_mean"])))
        d_std   = float(example.get("daily_sales_std",
                        example.get("flow_std",  ctx["flow_std"])))
        lag     = float(example.get("current_lead_time",
                        example.get("lag", ctx["lag"])))
        old_buf = float(example.get("old_safety_stock_coef",
                        example.get("old_coef", ctx["old_coef"])))

        eta  = d_std / max(d_mean, 1e-9)
        warn = (eta > eta_c) or (lag > tau_c)
        mult = max(1.0, (eta / eta_c) * (lag / tau_c)) if warn else 1.0

        return {
            "example_type":    "graph_invariant",
            "logia":           logia,
            "input": {
                ctx["flow_name"] + "_mean": d_mean,
                ctx["flow_name"] + "_std":  d_std,
                "lag_value":                lag,
                "old_buffer_coef":          old_buf,
                "_labels": {
                    "flow":   ctx["flow_label"],
                    "lag":    ctx["lag_label"],
                    "buffer": ctx["buffer_label"],
                },
            },
            "computed_cv":   round(eta, 4),
            "lag":           lag,
            "eta_critical":  eta_c,
            "tau_critical":  tau_c,
            "old_buffer":    round(old_buf * d_mean * lag, 2),
            "multiplier":    round(mult, 4),
            "new_buffer":    round(old_buf * d_mean * lag * mult, 2),
            "warning_triggered": warn,
        }

    elif t == "kuramoto":
        K     = float(example.get("coupling_K",   0.7))
        K_c   = float(example.get("K_c",          0.5))
        noise = float(example.get("noise_eta",    0.2))
        delay = float(example.get("delay_tau_hours",
                      example.get("delay_tau_ms",
                      example.get("delay_tau_days", 1.0))))
        warns = []
        if K < K_c:       warns.append(f"K={K:.3f} < K_c={K_c:.3f}")
        if noise > eta_c: warns.append(f"η={noise:.3f} > η_crit={eta_c:.3f}")
        if delay > tau_c: warns.append(f"τ={delay:.3f} > τ_crit={tau_c:.3f}")
        return {
            "example_type": "kuramoto",
            "logia":        logia,
            "input":        example,
            "K_above_Kc":   K > K_c,
            "noise_ok":     noise <= eta_c,
            "delay_ok":     delay <= tau_c,
            "warnings":     warns,
            "stable":       len(warns) == 0,
            "warning_triggered": len(warns) > 0,
        }

    elif t == "delay":
        K     = float(example.get("coupling_K",   0.3))
        noise = float(example.get("noise_eta",    0.2))
        delay = float(example.get("delay_tau",    1.0))
        K_min = float(ct.get("K_min", 0.1))
        m_n   = 1 - noise / max(eta_c, 1e-9)
        m_d   = 1 - delay / max(tau_c, 1e-9)
        m_k   = (K - K_min) / max(K_min, 1e-9)
        margin = min(m_n, m_d, m_k)
        return {
            "example_type":     "delay",
            "logia":            logia,
            "input":            example,
            "stability_margin": round(margin, 4),
            "noise_margin":     round(m_n, 4),
            "delay_margin":     round(m_d, 4),
            "coupling_margin":  round(m_k, 4),
            "warning_triggered": margin < 0.2,
        }

    elif t == "ising":
        import math
        K     = float(example.get("coupling_K",    0.8))
        T     = float(example.get("T_temperature", 0.9))
        noise = float(example.get("noise_eta",     0.15))
        tau   = float(example.get("tau_relax",
                      example.get("delay_tau", 1.0)))

        T_c = K
        try:
            m_val = math.tanh(K / max(T, 0.01))
        except Exception:
            m_val = 0.0
        order_noisy = max(0.0, abs(m_val) - noise * 0.3)

        warns = []
        if T >= T_c:       warns.append(f"T={T:.3f} ≥ T_c={T_c:.3f}: неупорядоченная фаза")
        if noise > eta_c:  warns.append(f"η={noise:.3f} > η_crit={eta_c:.3f}")
        if tau > tau_c:    warns.append(f"τ={tau:.3f} > τ_crit={tau_c:.3f}")

        stable = len(warns) == 0 and order_noisy > 0.3
        return {
            "example_type":    "ising",
            "logia":           logia,
            "input":           example,
            "T_c_approx":      round(T_c, 4),
            "order_parameter": round(order_noisy, 4),
            "phase":           "ordered" if T < T_c else "disordered",
            "K_above_Kc":      True,
            "noise_ok":        noise <= eta_c,
            "relax_ok":        tau <= tau_c,
            "warnings":        warns,
            "stable":          stable,
            "warning_triggered": len(warns) > 0,
        }

    elif t == "percolation":
        p_crit = float(
            ct.get("p_crit") or
            example.get("p_crit") or
            0.37
        )
        p     = float(example.get("p_measured",       0.52))
        K     = float(example.get("K_connectivity",   0.48))
        noise = float(example.get("noise_eta",        0.38))
        tau   = float(example.get("tau_lag_months",
                      example.get("monitoring_period_years", 6.8)))
        above = p > p_crit
        cascade_risk = max(0.0, (p - p_crit) / (1 - p_crit)) if above else 0.0
        warns = []
        if above:         warns.append(f"p={p:.3f} > p_crit={p_crit:.3f}: каскадный режим")
        if noise > eta_c: warns.append(f"η={noise:.3f} > η_crit={eta_c:.3f}")
        if tau > tau_c:   warns.append(f"τ={tau:.3f} > τ_crit={tau_c:.3f}")
        compound = round(cascade_risk * max(1.0, noise / max(eta_c, 1e-9)), 3)
        return {
            "example_type":    "percolation",
            "logia":           logia,
            "input": {
                **example,
                "p_crit": p_crit,
            },
            "p_crit":          p_crit,
            "above_threshold": above,
            "cascade_risk":    round(cascade_risk, 4),
            "compound_risk":   compound,
            "noise_ok":        noise <= eta_c,
            "warnings":        warns,
            "stable":          not above and compound < 0.3,
            "warning_triggered": len(warns) > 0,
        }

    elif t == "lotka_volterra":
        K     = float(example.get("interaction",     0.5))
        noise = float(example.get("noise_eta",       0.2))
        tau   = float(example.get("tau_cycle",
                      example.get("period_days", 10.0)))
        K_min = float(ct.get("K_min", 0.2))
        warns = []
        if K > K_min + 0.3:  warns.append(f"K={K:.3f}: конкуренция подавляет коэксистенцию")
        if noise > eta_c:    warns.append(f"η={noise:.3f} > η_crit={eta_c:.3f}")
        if tau > tau_c:      warns.append(f"τ={tau:.3f} > τ_crit={tau_c:.3f}")
        return {
            "example_type":    "lotka_volterra",
            "logia":           logia,
            "input":           example,
            "interaction_K":   K,
            "warnings":        warns,
            "stable":          len(warns) == 0,
            "warning_triggered": len(warns) > 0,
        }

    return {
        "error":     f"unknown example_data type: {t}",
        "supported": ["graph_invariant", "kuramoto", "delay", "ising", "percolation", "lotka_volterra"],
    }


# ══════════════════════════════════════════════════════════
# ПРОВЕРКА РЕАЛЬНЫХ ПАРАМЕТРОВ АРТЕФАКТА
# ══════════════════════════════════════════════════════════

def _calculate_with_artifact_params(model: Dict, flat: Dict, thresholds: Dict) -> Dict:
    """
    Проверяет РЕАЛЬНЫЕ параметры артефакта против порогов модели.
    Вычисляет risk_multiplier — во сколько раз буфер/запас нужно увеличить.
    """
    K     = flat.get("K",   0.35)
    eta   = flat.get("eta", 0.2)
    tau   = flat.get("tau", 0.5)
    eta_c = thresholds["eta_critical"]
    tau_c = thresholds["tau_robustness"]
    K_min = float(model.get("critical_thresholds", {}).get("K_min", 0.0))

    warns = []
    if K_min > 0 and K < K_min:
        warns.append(
            f"K={K:.3f} < K_min={K_min:.3f} "
            f"(артефакт ниже порога связи модели)"
        )
    if eta > eta_c:
        warns.append(
            f"η={eta:.3f} > η_crit={eta_c:.3f} "
            f"(шум превышает критический порог)"
        )
    if tau > tau_c:
        warns.append(
            f"τ={tau:.3f} > τ_crit={tau_c:.3f} "
            f"(задержка превышает критический порог)"
        )

    eta_ratio = eta / max(eta_c, 1e-9)
    tau_ratio = tau / max(tau_c, 1e-9)
    K_ratio   = K_min / max(K, 1e-9) if K_min > 0 else 0.0

    over_ratios = [r for r in [eta_ratio, tau_ratio, K_ratio] if r > 1.0]
    risk_multiplier = round(max(over_ratios), 3) if over_ratios else 1.0

    margin_eta = round(max(0.0, (eta_c - eta) / max(eta_c, 1e-9)), 3)
    margin_tau = round(max(0.0, (tau_c - tau) / max(tau_c, 1e-9)), 3)
    margin_K   = round(max(0.0, (K - K_min) / max(K_min, 1e-9)), 3) if K_min > 0 else None

    return {
        "example_type":     "artifact_params",
        "is_synthetic":     False,
        "artifact_K":       round(K,   4),
        "artifact_eta":     round(eta, 4),
        "artifact_tau":     round(tau, 4),
        "model_K_min":      K_min,
        "model_eta_crit":   eta_c,
        "model_tau_crit":   tau_c,
        "K_ok":             K >= K_min if K_min > 0 else True,
        "eta_ok":           eta <= eta_c,
        "tau_ok":           tau <= tau_c,
        "warnings":         warns,
        "stable":           len(warns) == 0,
        "warning_triggered": len(warns) > 0,
        "risk_multiplier":  risk_multiplier,
        "margins": {
            "eta":  margin_eta,
            "tau":  margin_tau,
            **({"K": margin_K} if margin_K is not None else {}),
        },
    }


def _compute_roi_estimate(
    artifact_calc: Dict,
    model: Dict,
    thresholds: Dict,
    stability_score: float = 0.5,
) -> str:
    """
    Вычисляет оценку снижения риска ТОЛЬКО на основе реальных данных
    артефакта (artifact_calc), не синтетического примера.

    Алгоритм:
      min_margin = min запас до критических порогов по всем параметрам
      base_potential зависит от stability_score
      potential = min_margin * base_potential → округляем до 5%
    """
    margins_dict = artifact_calc.get("margins", {})

    if not margins_dict:
        return "Недостаточно данных для оценки ROI"

    margins = [v for v in margins_dict.values() if v is not None]
    if not margins:
        return "Система на пороге нестабильности — ROI не определён"

    min_margin = min(margins)

    if stability_score >= 0.9:
        base = 0.30
    elif stability_score >= 0.7:
        base = 0.25
    elif stability_score >= 0.5:
        base = 0.20
    else:
        base = 0.10

    potential = min_margin * base

    if potential <= 0.005:
        risk_mult = artifact_calc.get("risk_multiplier", 1.0)
        if risk_mult > 1.0:
            return (
                f"Система за критическим порогом (множитель риска ×{risk_mult:.2f}) — "
                f"сначала стабилизировать параметры"
            )
        return "Система на пороге — профилактический мониторинг"

    lo = max(5, int(potential * 100 / 5) * 5)
    hi = lo + 5

    risk_mult = artifact_calc.get("risk_multiplier", 1.0)
    mult_str = f" | множитель риска ×{risk_mult:.2f}" if risk_mult > 1.0 else ""
    return f"Снижение риска каскадных отказов на {lo}–{hi}%{mult_str}"


def _build_calculation_summary(calculation: Dict, artifact_calc: Dict, thresholds: Dict) -> str:
    """
    Возвращает строку-резюме для UI (поле calculation_summary).
    UI ожидает строку, не dict.
    """
    risk_mult = artifact_calc.get("risk_multiplier", 1.0)
    stable    = artifact_calc.get("stable", True)
    warns     = artifact_calc.get("warnings", [])
    sc        = thresholds.get("stability_score", 0.0)

    if not stable and warns:
        short_warns = "; ".join(warns[:2])
        return (
            f"⚠ Реальные параметры за порогом: {short_warns}. "
            f"Множитель риска ×{risk_mult:.2f}. "
            f"stability={sc:.2f}"
        )
    return (
        f"✓ Параметры артефакта в норме. "
        f"Множитель риска ×{risk_mult:.2f}. "
        f"stability={sc:.2f}"
    )


def _build_verdict(
    model: Dict,
    calculation: Dict,
    artifact_calc: Dict,
    resonance: float,
    thresholds: Dict,
    flat: Dict,
) -> Dict:
    program = (model.get("programs") or ["target_system"])[0]
    warnings: List[str] = []

    flat = flat or {}
    # Форматируем blind_spot с реальными порогами модели
    ct_bd = model.get("critical_thresholds", {})
    blind_spot_formatted = _format_blind_spot(
        template  = model.get("blind_spot_template") or "—",
        eta_crit  = thresholds.get("eta_critical", ct_bd.get("eta_max", 0.5)),
        tau_crit  = thresholds.get("tau_robustness", ct_bd.get("tau_max", 5.0)),
        p_crit    = ct_bd.get("p_crit", 0.37),
        K_min     = ct_bd.get("K_min", 0.0),
        T_crit    = ct_bd.get("T_crit", 0.0),
        p         = flat.get("p", 0.5),
    )

    if calculation.get("warning_triggered"):
        warnings.append("Синтетический пример модели показал возможную нестабильность.")
    if artifact_calc.get("warning_triggered"):
        warnings.append("Параметры артефакта приближаются к критическому порогу.")
    if thresholds.get("stability_score", 1.0) < 0.5:
        warnings.append("Низкий stability_score артефакта.")
    if not thresholds.get("survival_verified", True):
        warnings.append("Выживаемость артефакта не подтверждена.")
    if resonance < 0.65:
        warnings.append("Низкая resonance — совпадение модели слабое.")

    if warnings:
        verdict_text = "Применимо как расширение"
        summary = " ".join(warnings[:2])
    else:
        verdict_text = "Применимо, мониторинг"
        summary = (
            f"Артефакт резонирует с моделью «{model.get('name', 'N/A')}" 
            f"({model.get('logia', '—')}, resonance={resonance:.2f})."
        )

    biz_rec = (
        "Рекомендуется усиленный мониторинг и проверка порогов."
        if warnings else "Поддерживающий мониторинг полезен для устойчивости системы."
    )

    return {
        "verdict": verdict_text,
        "for_developer": {
            "action":         f"Проверить адаптацию в {program} и настройки порогов",
            "code_reference": "adaptation.code_snippet",
            "new_config_params": {
                "eta_critical":   thresholds.get("eta_critical"),
                "tau_robustness": thresholds.get("tau_robustness"),
            },
            "artifact_warnings": artifact_calc.get("warnings", []),
        },
        "for_business": {
            "summary":         (
                f"Артефакт резонирует с моделью «{model.get('name')}» "
                f"({model.get('logia')}, resonance={resonance:.2f})."
            ),
            "blind_spot":      blind_spot_formatted,
            "recommendation":  biz_rec,
            "stability_score": thresholds.get("stability_score", "—"),
            "estimated_roi":   _compute_roi_estimate(
                                   artifact_calc=artifact_calc,
                                   model=model,
                                   thresholds=thresholds,
                                   stability_score=thresholds.get("stability_score", 0.5),
                               ),
        },
    }


# ══════════════════════════════════════════════════════════════════
# ОСНОВНОЙ КЛАСС
# ══════════════════════════════════════════════════════════

class MGAPMatcher:
    def __init__(self, registry_path: str = "mgap_registry.json", artifacts_dir: str = "artifacts"):
        self.registry_path = Path(registry_path)
        self.artifacts_dir = Path(artifacts_dir)
        self.registry = self._load_registry()
        self.llm = self._try_load_llm()

    def _load_registry(self) -> List[Dict]:
        if not self.registry_path.exists():
            logger.warning(f"Registry not found: {self.registry_path}")
            return []
        try:
            data = json.loads(self.registry_path.read_text(encoding="utf-8"))
            models = data.get("models", [])
            logger.info(
                f"MGAPMatcher: loaded {len(models)} models "
                f"({', '.join(data.get('math_types_covered', []))})"
            )
            return models
        except Exception as e:
            logger.error(f"Failed to load registry: {e}")
            return []

    def _try_load_llm(self):
        """
        Упрощённая загрузка LLMClient. Выбор модели делегирован MeshLLM.
        """
        try:
            from llm_client_v_4 import LLMClient
            from api_usage_tracker import tracker
            providers = tracker.get_providers_for_role("generator")
            if providers:
                logger.info(
                    f"MGAPMatcher LLM: primary provider = "
                    f"{providers[0].label} ({providers[0].provider})"
                )
            else:
                logger.warning("MGAPMatcher: no generator providers in tracker")
            return LLMClient()
        except Exception as e:
            logger.warning(f"MGAPMatcher: LLMClient unavailable — {e}")
            return None

    def _llm_generate(self, prompt: str, purpose: str = "mgap") -> tuple[str, str]:
        """
        Обёртка вокруг LLMClient.generate() с логированием модели.
        Возвращает (text, model_name).
        """
        if not self.llm:
            return "", "none"
        try:
            text, model = self.llm.generate(prompt)
            if text and not text.startswith("[Generator error]"):
                logger.info(f"MGAPMatcher [{purpose}]: ✓ {model.split('/')[-1]}")
                return text, model
            logger.warning(f"MGAPMatcher [{purpose}]: LLM failed — {text[:80]}")
            return "", model
        except Exception as e:
            logger.warning(f"MGAPMatcher [{purpose}]: exception — {e}")
            return "", "error"

    def _load_artifact(self, artifact_id: str) -> Optional[Dict]:
        for base in [self.artifacts_dir, Path(".")]:
            for name in [f"{artifact_id}.json", f"{artifact_id}.hyx-portal.json"]:
                p = base / name
                if p.exists():
                    try:
                        return json.loads(p.read_text(encoding="utf-8"))
                    except Exception:
                        pass
        return None

    def match_artifact(
        self,
        artifact_id: str,
        top_k: int = 3,
        math_type_only: bool = False,
        model_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        artifact = self._load_artifact(artifact_id)
        if not artifact:
            return [{"error": f"Artifact '{artifact_id}' not found",
                     "artifact_id": artifact_id}]

        four_d = _extract_art_four_d(artifact)
        if not four_d:
            return [{"error": "No four_d_matrix — run migrate_to_v42.py first",
                     "artifact_id": artifact_id}]

        sim     = _extract_art_sim(artifact)
        ver     = artifact.get("data", {}).get("ver", {})
        flat    = _flat_4d(four_d)
        art_math = _norm_math_type(flat["model"])
        art_vec  = _art_vector(four_d)

        candidates = self.registry
        if model_id:
            candidates = [m for m in candidates if m["id"] == model_id]
        if math_type_only:
            candidates = [m for m in candidates
                          if _norm_math_type(m.get("math_type", "")) == art_math]
        if not candidates:
            return [{"error": f"No matching models (math_type={art_math})",
                     "artifact_id": artifact_id}]

        scored: List[Tuple[float, Dict]] = []
        for model in candidates:
            res = _compute_resonance(art_vec, model, art_math)
            scored.append((res, model))
        scored.sort(key=lambda x: -x[0])

        results = []
        for resonance, model in scored[:top_k]:
            thresholds = _extract_thresholds(sim, ver, model)
            match = self._build_match(
                artifact_id=artifact_id,
                artifact=artifact,
                four_d=four_d,
                flat=flat,
                thresholds=thresholds,
                art_math=art_math,
                model=model,
                resonance=resonance,
            )
            results.append(match)
        return results

    def _build_match(
        self,
        artifact_id: str,
        artifact: Dict,
        four_d: Dict,
        flat: Dict,
        thresholds: Dict,
        art_math: str,
        model: Dict,
        resonance: float,
    ) -> Dict[str, Any]:
        math_match  = _norm_math_type(model.get("math_type", "")) == art_math
        translation = self._translate_params(flat, thresholds, model)
        ct_for_blind = model.get("critical_thresholds", {})
        # Подставляем пороги конкретной модели (не жёсткие числа из шаблона)
        _eta_c  = thresholds["eta_critical"]      # из stress_test артефакта или ct модели
        _tau_c  = thresholds["tau_robustness"]
        _p_crit = ct_for_blind.get("p_crit", 0.37)
        _K_min  = ct_for_blind.get("K_min", 0.0)
        _T_crit = ct_for_blind.get("T_crit", _K_min if _K_min else 1.0)

        raw_blind = _format_blind_spot(
            template=model.get("blind_spot_template") or "",
            eta_crit=_eta_c,
            tau_crit=_tau_c,
            p_crit=_p_crit,
            K_min=_K_min,
            T_crit=_T_crit,
            p=flat.get("p", 0.5),
        )
        # blind_spot через LLM (Groq/Mash first)
        blind_spot   = self._improve_blind_spot(raw_blind, model)
        code_snippet = _generate_code(model, thresholds, flat)
        calculation  = _calculate_example(model, thresholds)
        artifact_calc    = _calculate_with_artifact_params(model, flat, thresholds)
        verdict      = _build_verdict(model, calculation, artifact_calc, resonance, thresholds, flat)

        # v4.5.1: опциональный LLM-анализ совместимости
        llm_analysis = self._llm_analyze_match(artifact, model, resonance, flat, thresholds)

        gen      = artifact.get("data", {}).get("gen", {})
        archivist = artifact.get("archivist") or {}

        result = {
            "artifact_id":    artifact_id,
            "model_id":       model.get("id"),
            "model_name":     model.get("name"),
            "logia":          model.get("logia"),
            "industry":       model.get("industry"),
            "programs":       model.get("programs", []),
            "disc_code":      model.get("disc_code"),
            "sector_code":    model.get("sector_code"),
            "resonance":      resonance,
            "resonance_tier": (
                "высокий" if resonance >= 0.8 else
                "средний" if resonance >= 0.65 else
                "низкий"
            ),
            "risk_multiplier":  artifact_calc.get("risk_multiplier", 1.0),
            "math_type_match": math_match,
            "artifact_summary": {
                "domain":           artifact.get("data", {}).get("domain", "—"),
                "hypothesis":       gen.get("hypothesis", "")[:120],
                "math_type":        art_math,
                "stability_score":  thresholds["stability_score"],
                "survival_verified": thresholds["survival_verified"],
                "novelty":          archivist.get("novelty", "—"),
            },
            "thresholds":    thresholds,
            "translation":   translation,
            "blind_spot":    blind_spot,
            "adaptation": {
                "formula":      model.get("math_adaptation_formula", "—"),
                "code_snippet": code_snippet,
                "programs":     model.get("programs", []),
            },
            "calculation":        calculation,
            "calculation_summary": _build_calculation_summary(
                calculation, artifact_calc, thresholds
            ),
            "artifact_check":     artifact_calc,
            "verdict":            verdict,
            "generated_at":  __import__("datetime").datetime.utcnow().isoformat() + "Z",
        }
        if llm_analysis:
            result["llm_analysis"] = llm_analysis
        return result

    def _translate_params(self, flat: Dict, thresholds: Dict, model: Dict) -> Dict:
        tmap = model.get("translation_map") or {}
        mt   = _norm_math_type(model.get("math_type", ""))

        if mt == "ising":
            key_params = [("T", flat["T"]), ("K", flat["K"]), ("eta", flat["eta"])]
        elif mt == "percolation":
            key_params = [("p",   flat["p"]),
                          ("K",   flat["K"]),
                          ("eta", flat["eta"])]
            if "tau" in tmap:
                key_params.append(("tau", flat["tau"]))
        elif mt == "lotka_volterra":
            key_params = [("K",   flat["K"]),
                          ("tau", flat["tau"]),
                          ("eta", flat["eta"])]
        elif mt in ("delay", "delay_ode"):
            key_params = [("tau", flat["tau"]),
                          ("K",   flat["K"]),
                          ("eta", flat["eta"])]
        else:
            key_params = [("tau", flat["tau"]),
                          ("K",   flat["K"]),
                          ("eta", flat["eta"])]

        result: Dict = {}
        for key, val in key_params:
            if key in tmap:
                entry = tmap[key]
                result[entry["industry_term"]] = {
                    "math_param":    key,
                    "value":         round(val, 4),
                    "description":   entry.get("description", ""),
                    "typical_range": entry.get("typical_values", "—"),
                }
            else:
                result[key] = {"math_param": key, "value": round(val, 4)}

        result["_thresholds"] = {
            "eta_critical":   thresholds["eta_critical"],
            "tau_robustness": thresholds["tau_robustness"],
        }
        return result

    def _improve_blind_spot(self, template: str, model: Dict) -> str:
        """
        v4.6: жёсткий шаблон. Если template уже содержит числа из critical_thresholds —
        возвращаем как есть. LLM вызываем только для коротких шаблонов без чисел.
        Запрещаем LLM добавлять свой текст — только подставлять отраслевое последствие.
        """
        if not template:
            return template

        import re as _re
        has_numbers = bool(_re.search(r'\d+\.?\d*', template))
        if has_numbers and len(template) >= 60:
            return template

        if not self.llm:
            return template

        from retry_manager import retry_manager

        mt    = _norm_math_type(model.get("math_type", ""))
        logia = model.get("logia", "неизвестная отрасль")
        name  = model.get("name", "")
        ct    = model.get("critical_thresholds", {})

        numbers_context = []
        if ct.get("eta_max") is not None:
            numbers_context.append(f"η_max={ct['eta_max']}")
        if ct.get("tau_max") is not None:
            numbers_context.append(f"τ_max={ct['tau_max']}")
        if ct.get("p_crit") is not None:
            numbers_context.append(f"p_crit={ct['p_crit']}")
        if ct.get("K_min") is not None:
            numbers_context.append(f"K_min={ct['K_min']}")
        if ct.get("T_crit") is not None:
            numbers_context.append(f"T_crit={ct['T_crit']}")
        numbers_str = ", ".join(numbers_context) if numbers_context else "см. реестр"

        fewshot_map = {
            "percolation": (
                "Пример: «Стандартные модели эпидемиологии не отслеживают "
                "p_crit=0.572 и η_max=0.5: при p > p_crit происходит "
                "взрывной рост заражений, который не прогнозируется SIR-моделью.»"
            ),
            "kuramoto": (
                "Пример: «Стандартные алгоритмы не отслеживают τ_max=2.3 "
                "и η_max=0.38: превышение τ в Brian2 переводит нейронную сеть "
                "в режим десинхронизации с потерей когерентности.»"
            ),
            "graph_invariant": (
                "Пример: «Стандартные WMS не учитывают CV>0.5 и lag>4.5 дней: "
                "при превышении страховой запас SAP EWM занижается на 20–50%.»"
            ),
            "delay": (
                "Пример: «Стандартные модели не отслеживают τ_max=5.5 и K_min=0.3: "
                "при K·τ > π/2 система в Dynare входит в колебательную неустойчивость.»"
            ),
            "ising": (
                "Пример: «Стандартные расчёты не отслеживают T_crit=0.71 "
                "и η_max=0.4: при T ≥ T_crit в LAMMPS кристалл переходит "
                "в неупорядоченную фазу с потерей дальнего порядка.»"
            ),
        }
        fewshot = fewshot_map.get(mt, fewshot_map.get("graph_invariant", ""))

        prompt = (
            f"Дополни описание слепой зоны для модели «{name}» "
            f"(отрасль: {logia}, math_type: {mt}).\n\n"
            f"{fewshot}\n\n"
            f"ТРЕБОВАНИЯ (строго):\n"
            f"  1. Используй ВСЕ числа из списка: {numbers_str}\n"
            f"  2. Добавь ОДНО конкретное последствие для отрасли «{logia}»\n"
            f"  3. Упомяни конкретную программу: "
            f"{', '.join((model.get('programs') or ['целевую систему'])[:2])}\n"
            f"  4. НЕ добавляй общих слов «система», «процесс», «механизм» "
            f"без конкретного субъекта\n"
            f"  5. Ответ — ТОЛЬКО текст описания, один абзац, без пояснений\n\n"
            f"Исходный шаблон:\n{template}"
        )

        def _call():
            text, m = self._llm_generate(prompt, purpose="blind_spot")
            return text, m

        result = retry_manager.call_with_retry(
            func=_call,
            validator=retry_manager.validator_field(min_len=50),
            context="mgap/blind_spot",
        )

        if result and result.value:
            improved, _ = result.value
            improved = improved.strip()
            garbage_markers = (
                "инструкц", "создание улучшенного", "вот описание",
                "вот улучшенное", "машинное обучение", "градиентный"
            )
            if (len(improved) > 600
                    or any(m in improved.lower() for m in garbage_markers)):
                logger.warning("LLM blind_spot returned invalid text, keeping original template")
                return template
            return improved
        return template

    def _llm_analyze_match(
        self,
        artifact: Dict,
        model: Dict,
        resonance: float,
        flat: Dict,
        thresholds: Dict,
    ) -> Optional[Dict]:
        """
        v4.6: few-shot в промпте, обязательные числовые ссылки,
        запрет пересказа гипотезы, предсказание для конкретной программы.
        """
        if not self.llm or resonance < 0.5:
            return None

        gen        = artifact.get("data", {}).get("gen", {})
        hypothesis = gen.get("hypothesis", "")[:250]
        if not hypothesis:
            return None

        mt         = _norm_math_type(model.get("math_type", ""))
        logia      = model.get("logia", "")
        name       = model.get("name", "")
        progs      = (model.get("programs") or ["целевую систему"])[:2]
        prog1      = progs[0]
        progs_str  = ", ".join(progs)
        eta_c      = thresholds["eta_critical"]
        tau_c      = thresholds["tau_robustness"]
        K_val      = flat.get("K",   0.35)
        eta_val    = flat.get("eta", 0.2)
        tau_val    = flat.get("tau", 0.5)
        p_val      = flat.get("p",   0.65)
        art_domain = artifact.get("data", {}).get("domain", "?")
        ct         = model.get("critical_thresholds", {})
        K_min      = float(ct.get("K_min", 0.0))
        p_crit     = float(ct.get("p_crit", 0.0))

        if mt == "percolation":
            params_block = (
                f"  p={p_val:.3f}  K={K_val:.3f}  η={eta_val:.3f}\n"
                f"  p_crit={p_crit:.3f}  η_crit={eta_c:.3f}  τ_crit={tau_c:.3f}"
            )
            risk_hint = (
                f"Какой из порогов p_crit={p_crit:.3f} или η_crit={eta_c:.3f} "
                f"ближе к нарушению? Что произойдёт в {prog1}?"
            )
        elif mt == "kuramoto":
            params_block = (
                f"  K={K_val:.3f}  K_c≈{flat.get('K_c', 0.5):.3f}  "
                f"η={eta_val:.3f}  τ={tau_val:.3f}\n"
                f"  η_crit={eta_c:.3f}  τ_crit={tau_c:.3f}  K_min={K_min:.3f}"
            )
            risk_hint = (
                f"Что ближе к порогу: K vs K_c, или η={eta_val:.3f} vs η_crit={eta_c:.3f}, "
                f"или τ={tau_val:.3f} vs τ_crit={tau_c:.3f}? Что случится в {prog1}?"
            )
        elif mt == "ising":
            T_crit = float(ct.get("T_crit", K_val))
            T_val  = flat.get("T", 1.0)
            params_block = (
                f"  T={T_val:.3f}  K={K_val:.3f}  η={eta_val:.3f}\n"
                f"  T_crit={T_crit:.3f}  η_crit={eta_c:.3f}"
            )
            risk_hint = (
                f"Упорядоченная фаза: T < T_crit={T_crit:.3f}. "
                f"Насколько T={T_val:.3f} близко к порогу? Что в {prog1}?"
            )
        elif mt in ("delay", "delay_ode"):
            params_block = (
                f"  K={K_val:.3f}  τ={tau_val:.3f}  η={eta_val:.3f}\n"
                f"  K_min={K_min:.3f}  τ_crit={tau_c:.3f}  η_crit={eta_c:.3f}"
            )
            risk_hint = (
                f"Как близки K={K_val:.3f} и K_min={K_min:.3f}, или τ={tau_val:.3f} "
                f"и τ_crit={tau_c:.3f}? Как это повлияет на {prog1}?"
            )
        elif mt == "lotka_volterra":
            params_block = (
                f"  K={K_val:.3f}  η={eta_val:.3f}  τ={tau_val:.3f}\n"
                f"  K_min={K_min:.3f}  η_crit={eta_c:.3f}  τ_crit={tau_c:.3f}"
            )
            risk_hint = (
                f"Насколько взаимодействие K={K_val:.3f} превышает K_min={K_min:.3f}, "
                f"и что произойдёт в {prog1} при τ={tau_val:.3f}?"
            )
        else:
            params_block = (
                f"  K={K_val:.3f}  η={eta_val:.3f}  τ={tau_val:.3f}\n"
                f"  η_crit={eta_c:.3f}  τ_crit={tau_c:.3f}"
            )
            risk_hint = (
                f"Что важнее для {prog1}: η={eta_val:.3f} vs η_crit={eta_c:.3f} или "
                f"τ={tau_val:.3f} vs τ_crit={tau_c:.3f}?"
            )

        fewshot = (
            "ПРИМЕР ХОРОШЕГО ОТВЕТА:\n"
            '{"why_applicable": "Инвариант (синхронизация нейронов, domain=neuroscience) '
            'резонирует с моделью нейронной синхронизации: оба kuramoto, '
            'K=0.874 > K_c=0.508 — система за критическим порогом, '
            'η=0.22 < η_crit=0.38 — шум не разрушает когерентность.", '
            '"main_risk": "τ=1.479 близко к τ_crit=2.3 (запас 22%): '
            'рост задержек в Brian2 на 50% переведёт систему за порог — '
            'desync и потеря паттернов.", '
            '"dev_action": "В Brian2 перед simulate(): '
            'mgap_stability_monitor(K=0.874, eta=0.22, tau=1.479). '
            'Добавить assert coupling_strength > 0.666.", '
            '"confidence": 0.82}'
        )

        prompt = (
            f"Ты MGAP-аналитик. Оцени применимость инварианта к отраслевой модели.\n\n"
            f"{fewshot}\n\n"
            f"───\n"
            f"Инвариант (домен: {art_domain}): {hypothesis}\n"
            f"Модель: «{name}» ({logia}), math_type={mt}\n"
            f"Программы: {progs_str}\n\n"
            f"Числовые параметры артефакта:\n{params_block}\n\n"
            f"Вопрос для main_risk: {risk_hint}\n\n"
            f"ПРАВИЛА ОТВЕТА:\n"
            f"  1. why_applicable: назови конкретные параметры с числами и объясни "
            f"почему они совместимы с моделью {name}.\n"
            f"  2. main_risk: укажи КОНКРЕТНЫЙ порог ближайший к нарушению "
            f"(с числами) и что произойдёт в {prog1}.\n"
            f"  3. dev_action: конкретный вызов функции/метода в {prog1} "
            f"с реальными значениями параметров.\n"
            f"  4. НЕ пересказывай гипотезу. Анализируй ПРИМЕНИМОСТЬ.\n"
            f"  5. НЕ используй слова «система», «процесс», «механизм» "
            f"без конкретного субъекта из {logia}.\n"
            f"  6. Все числа из блока параметров ДОЛЖНЫ появиться в ответе.\n\n"
            f"Верни ТОЛЬКО JSON:\n"
            f'{{"why_applicable": "...", "main_risk": "...", '
            f'"dev_action": "...", "confidence": 0.0}}'
        )

        import re as _re

        def _call_analysis():
            text, model_used = self._llm_generate(prompt, purpose="match_analysis")
            return text, model_used

        from retry_manager import retry_manager
        result = retry_manager.call_with_retry(
            func=_call_analysis,
            validator=retry_manager.validator_field(min_len=50),
            context="mgap/match_analysis",
        )

        if not result or not result.value:
            return None

        text, model_used = result.value
        if not text:
            return None

        cleaned = _re.sub(r"```(?:json)?", "", text).replace("```", "").strip()
        match   = _re.search(r"\{[\s\S]*\}", cleaned)
        if not match:
            return None
        try:
            res = json.loads(match.group(0))
            res["_model"] = model_used.split("/")[-1] if model_used else "?"
            return res
        except Exception:
            return None

    def match_batch(
        self,
        top_k: int = 2,
        math_type_only: bool = True,
        min_resonance: float = 0.3,
    ) -> Dict[str, List[Dict]]:
        results: Dict[str, List[Dict]] = {}
        if not self.artifacts_dir.exists():
            return results
        for f in sorted(self.artifacts_dir.glob("*.json")):
            if f.stem == "invariant_graph" or ".hyx-portal" in f.name:
                continue
            art_id = f.stem
            try:
                matches = self.match_artifact(art_id, top_k=top_k,
                                               math_type_only=math_type_only)
                ok = [m for m in matches
                      if not m.get("error") and m.get("resonance", 0) >= min_resonance]
                if ok:
                    results[art_id] = ok
                    logger.info(f"MGAP batch: {art_id} → "
                                f"{[(m['model_id'], m['resonance']) for m in ok]}")
            except Exception as e:
                logger.warning(f"MGAP batch: {art_id} failed — {e}")
        return results

    def get_registry_summary(self) -> List[Dict]:
        return [
            {
                "id":        m["id"],
                "name":      m["name"],
                "logia":     m["logia"],
                "industry":  m["industry"],
                "math_type": m.get("math_type", "—"),
                "disc_code": m.get("disc_code"),
                "sector_code": m.get("sector_code"),
                "programs":  m.get("programs", []),
            }
            for m in self.registry
        ]


# ══════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════

def _cli():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    parser = argparse.ArgumentParser(description="HX-AM v4.5 MGAPMatcher CLI")
    parser.add_argument("--artifact",      type=str, default="")
    parser.add_argument("--model",         type=str, default="")
    parser.add_argument("--top_k",         type=int, default=3)
    parser.add_argument("--all_types",     action="store_true")
    parser.add_argument("--batch",         action="store_true")
    parser.add_argument("--registry",      action="store_true")
    parser.add_argument("--min_res",       type=float, default=0.3)
    parser.add_argument("--registry_path", type=str, default="mgap_registry.json")
    parser.add_argument("--artifacts_dir", type=str, default="artifacts")
    args = parser.parse_args()

    matcher = MGAPMatcher(registry_path=args.registry_path, artifacts_dir=args.artifacts_dir)

    if args.registry:
        print(json.dumps(matcher.get_registry_summary(), ensure_ascii=False, indent=2))
        return

    if args.batch:
        results = matcher.match_batch(
            top_k=args.top_k,
            math_type_only=not args.all_types,
            min_resonance=args.min_res,
        )
        print(json.dumps(results, ensure_ascii=False, indent=2))
        return

    if not args.artifact:
        parser.print_help()
        return

    results = matcher.match_artifact(
        artifact_id=args.artifact,
        top_k=args.top_k,
        math_type_only=not args.all_types,
        model_id=args.model or None,
    )
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    _cli()
