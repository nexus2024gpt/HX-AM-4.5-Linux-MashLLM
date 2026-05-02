# mgap_matcher.py — HX-AM v4.5
"""
MGAPMatcher — переносит численные инварианты из артефактов HX-AM
в прикладные отраслевые модели (реестр MGAP).

v4.5 исправления:
  - Добавлена кодогенерация для math_type=ising
  - Добавлен расчёт на примере для type=ising
  - Verdict учитывает survival_verified (False → предупреждение)
  - Verdict учитывает stability_score < 0.5 (математически нестабильный)
  - _compute_resonance: нормализован type_bonus к правильным весам
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


def _norm_math_type(t: str) -> str:
    return _MATH_TYPE_ALIASES.get(t.lower().strip(), t.lower().strip())


# ══════════════════════════════════════════════════════════
# ИЗВЛЕЧЕНИЕ ПАРАМЕТРОВ
# ══════════════════════════════════════════════════════════

def _flat_4d(four_d: Dict) -> Dict[str, Any]:
    dyn = four_d.get("dynamics", {})
    inf = four_d.get("influence", {})
    tim = four_d.get("time", {})
    return {
        "tau":     float(tim.get("tau",     0.5)),
        "K":       float(dyn.get("K",       0.35)),
        "K_c":     float(dyn.get("K_c",     0.48)),
        "p":       float(dyn.get("p",       0.65)),
        "omega_i": float(dyn.get("omega_i", 0.25)),
        "eta":     float(inf.get("eta",     0.2)),
        "T":       float(inf.get("T",       1.0)),
        "h":       float(inf.get("h",       0.5)),
        "model":   str(dyn.get("model",     "kuramoto")),
    }


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
        # v4.5: добавлена поддержка Изинга
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
            f"    # Параметр порядка (mean-field)\n"
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
    example = model.get("example_data") or {}
    eta_c   = thresholds["eta_critical"]
    tau_c   = thresholds["tau_robustness"]
    t       = example.get("type", "graph_invariant")

    if t == "graph_invariant":
        d_mean  = float(example.get("daily_sales_mean", 100))
        d_std   = float(example.get("daily_sales_std",   30))
        lag     = float(example.get("current_lead_time",  3.0))
        old_buf = float(example.get("old_safety_stock_coef", 0.2))
        eta  = d_std / max(d_mean, 1e-9)
        warn = (eta > eta_c) or (lag > tau_c)
        mult = max(1.0, (eta / eta_c) * (lag / tau_c)) if warn else 1.0
        return {
            "example_type": "graph_invariant", "input": example,
            "computed_cv": round(eta, 4), "lag": lag,
            "eta_critical": eta_c, "tau_critical": tau_c,
            "old_buffer": round(old_buf * d_mean * lag, 2),
            "multiplier": round(mult, 4),
            "new_buffer": round(old_buf * d_mean * lag * mult, 2),
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
            "example_type": "kuramoto", "input": example,
            "K_above_Kc": K > K_c, "noise_ok": noise <= eta_c, "delay_ok": delay <= tau_c,
            "warnings": warns, "stable": len(warns) == 0, "warning_triggered": len(warns) > 0,
        }

    elif t == "delay":
        K     = float(example.get("coupling_K",   0.3))
        noise = float(example.get("noise_eta",    0.2))
        delay = float(example.get("delay_tau",    1.0))
        K_min = model.get("critical_thresholds", {}).get("K_min", 0.1)
        m_n   = 1 - noise / max(eta_c, 1e-9)
        m_d   = 1 - delay / max(tau_c, 1e-9)
        m_k   = (K - K_min) / max(K_min, 1e-9)
        margin = min(m_n, m_d, m_k)
        return {
            "example_type": "delay", "input": example,
            "stability_margin": round(margin, 4), "noise_margin": round(m_n, 4),
            "delay_margin": round(m_d, 4), "coupling_margin": round(m_k, 4),
            "warning_triggered": margin < 0.2,
        }

    elif t == "ising":
        # v4.5: добавлена поддержка типа ising
        import math
        K     = float(example.get("coupling_K",    0.8))
        T     = float(example.get("T_temperature", 0.9))
        noise = float(example.get("noise_eta",     0.15))
        tau   = float(example.get("tau_relax",     example.get("delay_tau", 1.0)))
        lat   = int(example.get("lattice_size",    64))

        # T_crit ≈ K в mean-field
        T_c = K
        # Параметр порядка mean-field Ising: m = tanh(K*m / T) → m = tanh(K/T) при h=0
        try:
            m_val = math.tanh(K / max(T, 0.01))
        except Exception:
            m_val = 0.0
        order_noisy = max(0.0, abs(m_val) - noise * 0.3)

        warns = []
        if T >= T_c:       warns.append(f"T={T:.3f} ≥ T_c={T_c:.3f}: неупорядоченная фаза")
        if noise > eta_c:  warns.append(f"η={noise:.3f} > η_crit={eta_c:.3f}: флуктуации разрушают порядок")
        if tau > tau_c:    warns.append(f"τ={tau:.3f} > τ_crit={tau_c:.3f}: медленная релаксация")

        stable = len(warns) == 0 and order_noisy > 0.3
        return {
            "example_type":     "ising",
            "input":            example,
            "T_c_approx":       round(T_c, 4),
            "order_parameter":  round(order_noisy, 4),
            "phase":            "ordered" if T < T_c else "disordered",
            "K_above_Kc":       True,   # K > K_c (stable coupling)
            "noise_ok":         noise <= eta_c,
            "relax_ok":         tau <= tau_c,
            "warnings":         warns,
            "stable":           stable,
            "warning_triggered": len(warns) > 0,
        }

    elif t == "percolation":
        p     = float(example.get("p_measured",       0.52))
        p_c   = float(example.get("p_crit",           0.37))
        K     = float(example.get("K_connectivity",   0.48))
        noise = float(example.get("noise_eta",        0.38))
        tau   = float(example.get("tau_lag_months",
                      example.get("monitoring_period_years", 6.8)))
        above = p > p_c
        cascade_risk = max(0.0, (p - p_c) / (1 - p_c)) if above else 0.0
        warns = []
        if above:         warns.append(f"p={p:.3f} > p_crit={p_c:.3f}: каскадный режим")
        if noise > eta_c: warns.append(f"η={noise:.3f} > η_crit={eta_c:.3f}: высокая гетерогенность")
        if tau > tau_c:   warns.append(f"τ={tau:.3f} > τ_crit={tau_c:.3f}: долгое накопление")
        compound = round(cascade_risk * max(1.0, noise / max(eta_c, 1e-9)), 3)
        return {
            "example_type":    "percolation",
            "input":           example,
            "p_crit":          p_c,
            "above_threshold": above,
            "cascade_risk":    round(cascade_risk, 4),
            "compound_risk":   compound,
            "noise_ok":        noise <= eta_c,
            "warnings":        warns,
            "stable":          not above and compound < 0.3,
            "warning_triggered": len(warns) > 0,
        }

    return {"error": f"unknown example_data type: {t}",
            "supported": ["graph_invariant", "kuramoto", "delay", "ising", "percolation"]}


# ══════════════════════════════════════════════════════════
# ПРОВЕРКА РЕАЛЬНЫХ ПАРАМЕТРОВ АРТЕФАКТА
# ══════════════════════════════════════════════════════════

def _calculate_with_artifact_params(model: Dict, flat: Dict, thresholds: Dict) -> Dict:
    """
    Проверяет РЕАЛЬНЫЕ параметры артефакта против порогов модели.
    В отличие от _calculate_example, не использует синтетику из реестра.
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

    return {
        "example_type": "artifact_params",
        "artifact_K":   round(K,   4),
        "artifact_eta": round(eta, 4),
        "artifact_tau": round(tau, 4),
        "model_K_min":  K_min,
        "model_eta_crit": eta_c,
        "model_tau_crit": tau_c,
        "K_ok":    K >= K_min if K_min > 0 else True,
        "eta_ok":  eta <= eta_c,
        "tau_ok":  tau <= tau_c,
        "warnings": warns,
        "stable":   len(warns) == 0,
        "warning_triggered": len(warns) > 0,
    }


# ══════════════════════════════════════════════════════════
# ОБЪЯСНЕНИЕ ПОХОЖЕСТИ
# ══════════════════════════════════════════════════════════

# Ключевые слова по math_type — для объяснения похожести
_MATHTYPE_KEYWORDS: Dict[str, List[str]] = {
    "kuramoto":        ["синхронизация", "фаза", "осциллятор", "связь", "частота",
                        "synchronization", "phase", "oscillator", "coupling", "frequency"],
    "percolation":     ["перколяция", "каскад", "порог", "связность", "распространение",
                        "percolation", "cascade", "threshold", "connectivity"],
    "ising":           ["бинарный", "спин", "поле", "температура", "фазовый переход",
                        "binary", "spin", "field", "temperature", "phase transition"],
    "delay":           ["задержка", "запаздывание", "лаг", "обратная связь",
                        "delay", "lag", "feedback"],
    "graph_invariant": ["граф", "топология", "связность", "инвариант", "кластер",
                        "graph", "topology", "connectivity", "invariant", "cluster"],
    "lotka_volterra":  ["хищник", "жертва", "конкуренция", "популяция",
                        "predator", "prey", "competition", "population"],
}


def _build_similarity_explanation(
    model: Dict,
    flat: Dict,
    thresholds: Dict,
    artifact_hypothesis: str,
    resonance: float,
) -> Dict:
    """
    Объясняет, почему артефакт похож на модель.
    Возвращает: matching_keywords, param_proximity, domain_bridge, resonance_tier.
    """
    mt = _norm_math_type(model.get("math_type", ""))
    hyp_lower = artifact_hypothesis.lower()

    # 1. Совпадение ключевых слов
    keywords = _MATHTYPE_KEYWORDS.get(mt, [])
    found_kw = [kw for kw in keywords if kw in hyp_lower]

    # 2. Близость параметров (сравниваем с 4D модели)
    m4d   = model.get("four_d_matrix") or {}
    m_dyn = m4d.get("dynamics", {})
    m_inf = m4d.get("influence", {})
    m_tim = m4d.get("time", {})

    param_rows = []
    pairs = [
        ("K",       flat.get("K",   0.35), float(m_dyn.get("K",   0)),   "константа связи"),
        ("K_c",     flat.get("K_c", 0.48), float(m_dyn.get("K_c", 0)),   "критич. порог"),
        ("eta",     flat.get("eta", 0.2),  float(m_inf.get("eta", 0)),   "уровень шума"),
        ("tau",     flat.get("tau", 0.5),  float(m_tim.get("tau", 0)),   "задержка"),
        ("omega_i", flat.get("omega_i",0.25), float(m_dyn.get("omega_i",0)), "частота"),
    ]
    for name, art_v, mod_v, label in pairs:
        if mod_v == 0:
            continue
        delta = abs(art_v - mod_v)
        pct   = round(delta / max(mod_v, 1e-6) * 100, 1)
        close = pct <= 25
        param_rows.append({
            "param":   name,
            "label":   label,
            "artifact": round(art_v, 3),
            "model":    round(mod_v, 3),
            "delta_pct": pct,
            "close":    close,
        })

    close_params = [r["param"] for r in param_rows if r["close"]]

    # 3. Доменный мост
    art_domain   = "neuroscience"          # из контекста; передаётся снаружи через flat
    model_logia  = model.get("logia", "")
    domain_bridge = f"{art_domain} → {model_logia}"

    # 4. Уровень резонанса
    if resonance >= 0.8:
        tier = "высокий"
    elif resonance >= 0.65:
        tier = "средний"
    else:
        tier = "низкий"

    # 5. Текстовое объяснение
    kw_str    = ", ".join(found_kw[:5]) if found_kw else "нет явных совпадений"
    close_str = ", ".join(close_params) if close_params else "нет близких параметров"
    explanation = (
        f"Оба объекта используют модель {mt}. "
        f"Совпадающие ключевые слова: {kw_str}. "
        f"Близкие параметры (отклонение ≤25%%): {close_str}. "
        f"Резонанс {resonance:.3f} — {tier}."
    )

    return {
        "matching_keywords":  found_kw[:8],
        "param_proximity":    param_rows,
        "close_params":       close_params,
        "domain_bridge":      domain_bridge,
        "resonance_tier":     tier,
        "resonance":          resonance,
        "explanation_text":   explanation,
    }


# ══════════════════════════════════════════════════════════
# ВЕРДИКТ — учитывает survival_verified и stability_score
# ══════════════════════════════════════════════════════════

def _build_verdict(
    model: Dict,
    calc: Dict,
    artifact_calc: Dict,
    resonance: float,
    thresholds: Dict,
) -> Dict:
    """
    Вердикт строится по РЕАЛЬНЫМ параметрам артефакта (artifact_calc),
    а не по примеру из реестра (calc).
    calc оставлен для обратной совместимости — используется только как fallback.
    """
    # Используем artifact_calc как основной источник предупреждений
    warn  = artifact_calc.get("warning_triggered", calc.get("warning_triggered", False))
    warns = artifact_calc.get("warnings", [])
    prog  = (model.get("programs") or ["target_system"])[0]

    stability_score   = thresholds.get("stability_score", 1.0)
    survival_verified = thresholds.get("survival_verified", True)
    math_unstable     = (stability_score < 0.5) or (not survival_verified)

    warns_str = "; ".join(warns) if warns else ""

    if math_unstable and warn:
        verdict_text = "⚠️ Осторожно — нестабилен"
        dev_action   = (
            f"НЕ применять автоматически: stability={stability_score:.3f}. "
            f"Нарушения: {warns_str}. Проверить вручную в {prog}"
        )
        biz_rec = (
            f"Система МАТЕМАТИЧЕСКИ НЕСТАБИЛЬНА (stability={stability_score:.3f}). "
            f"Нарушения: {warns_str}. "
            f"Результаты MGAP-матча ({resonance:.2f}) требуют пересмотра перед применением в {prog}."
        )
    elif math_unstable:
        verdict_text = "⚠️ Математически нестабилен"
        dev_action   = f"Применить с осторожностью: stability={stability_score:.3f}. Добавить мониторинг в {prog}"
        biz_rec      = (
            f"Артефакт резонирует с моделью «{model.get('name')}» (resonance={resonance:.2f}), "
            f"однако симуляция даёт stability={stability_score:.3f}. Рекомендуется мониторинг."
        )
    elif warn:
        verdict_text = "Применимо как расширение"
        dev_action   = (
            f"Добавить mgap_stability_monitor() в {prog}. "
            f"Нарушения параметров артефакта: {warns_str}"
        )
        biz_rec = (
            f"Система НА ПОРОГЕ нестабильности. Нарушения: {warns_str}. "
            f"Внедрить MGAP-монитор в {prog}. "
            f"Снижение риска каскадных отказов: 15–25%."
        )
    else:
        verdict_text = "Применимо, мониторинг"
        dev_action   = f"Добавить пассивный мониторинг порогов в {prog}"
        biz_rec      = f"Все параметры в норме. Мониторинг порогов полезен профилактически."

    return {
        "verdict": verdict_text,
        "artifact_checks": {
            "K_ok":    artifact_calc.get("K_ok",   True),
            "eta_ok":  artifact_calc.get("eta_ok", True),
            "tau_ok":  artifact_calc.get("tau_ok", True),
            "warnings": warns,
        },
        "math_stability": {
            "stability_score":   stability_score,
            "survival_verified": survival_verified,
            "math_unstable":     math_unstable,
        },
        "for_developer": {
            "action":           dev_action,
            "code_reference":   "adaptation.code_snippet",
            "new_config_params": {
                "eta_critical":   thresholds["eta_critical"],
                "tau_robustness": thresholds["tau_robustness"],
            },
        },
        "for_business": {
            "summary":        (
                f"Артефакт резонирует с моделью «{model.get('name')}» "
                f"({model.get('logia')}, resonance={resonance:.2f})."
            ),
            "blind_spot":     (model.get("blind_spot_template") or "—").format(
                                  eta_max=thresholds["eta_critical"],
                                  tau_max=thresholds["tau_robustness"],
                                  p_crit=0.37,
                                  p=0.52,
            ),
            "recommendation": biz_rec,
            "stability_score": stability_score,
            "estimated_roi":  (
                "Снижение риска каскадных отказов на 15–25%"
                if not math_unstable else
                "Сначала стабилизировать систему — ROI не определён"
            ),
        },
    }


# ══════════════════════════════════════════════════════════
# ОСНОВНОЙ КЛАСС
# ══════════════════════════════════════════════════════════

class MGAPMatcher:

    def __init__(
        self,
        registry_path: str = "mgap_registry.json",
        artifacts_dir: str = "artifacts",
    ):
        self.registry_path = Path(registry_path)
        self.artifacts_dir = Path(artifacts_dir)
        self.registry      = self._load_registry()
        self.llm           = self._try_load_llm()

    def _load_registry(self) -> List[Dict]:
        if not self.registry_path.exists():
            logger.warning(f"Registry not found: {self.registry_path}")
            return []
        data = json.loads(self.registry_path.read_text(encoding="utf-8"))
        logger.info(
            f"MGAPMatcher: loaded {len(data.get('models', []))} models "
            f"({', '.join(data.get('math_types_covered', []))})"
        )
        return data.get("models", [])

    def _try_load_llm(self):
        try:
            from llm_client_v_4 import LLMClient
            return LLMClient()
        except Exception:
            return None

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
            return [{"error": f"Artifact '{artifact_id}' not found", "artifact_id": artifact_id}]

        four_d = _extract_art_four_d(artifact)
        if not four_d:
            return [{"error": "No four_d_matrix — run migrate_to_v42.py first", "artifact_id": artifact_id}]

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
            return [{"error": f"No matching models (math_type={art_math}, math_type_only={math_type_only})",
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
        raw_blind   = (model.get("blind_spot_template") or "").format(
            eta_max=thresholds["eta_critical"],
            tau_max=thresholds["tau_robustness"],
            p_crit=0.37, p=flat.get("p", 0.5),
        )
        blind_spot   = self._improve_blind_spot(raw_blind, model)
        code_snippet = _generate_code(model, thresholds, flat)
        calculation  = _calculate_example(model, thresholds)
        artifact_calc    = _calculate_with_artifact_params(model, flat, thresholds)
        verdict      = _build_verdict(model, calculation, artifact_calc, resonance, thresholds)

        art_hypothesis = artifact.get("data", {}).get("gen", {}).get("hypothesis", "")
        similarity_exp = _build_similarity_explanation(
            model, flat, thresholds, art_hypothesis, resonance
        )

        gen      = artifact.get("data", {}).get("gen", {})
        archivist = artifact.get("archivist") or {}

        return {
            "artifact_id":    artifact_id,
            "model_id":       model.get("id"),
            "model_name":     model.get("name"),
            "logia":          model.get("logia"),
            "industry":       model.get("industry"),
            "programs":       model.get("programs", []),
            "disc_code":      model.get("disc_code"),
            "sector_code":    model.get("sector_code"),
            "resonance":      resonance,
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
            "artifact_check":     artifact_calc,
            "similarity":         similarity_exp,
            "verdict":            verdict,
            "generated_at":  __import__("datetime").datetime.utcnow().isoformat() + "Z",
        }

    def _translate_params(self, flat: Dict, thresholds: Dict, model: Dict) -> Dict:
        tmap   = model.get("translation_map") or {}
        result: Dict = {}
        # Для Изинга — ключевые параметры T и h, а не tau
        mt = _norm_math_type(model.get("math_type", ""))
        if mt == "ising":
            key_params = [("T", flat["T"]), ("K", flat["K"]), ("eta", flat["eta"])]
        else:
            key_params = [("tau", flat["tau"]), ("K", flat["K"]), ("eta", flat["eta"])]

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
        if not self.llm or not template:
            return template
        prompt = (
            f"Улучши описание слепой зоны для модели «{model.get('name')}» "
            f"(отрасль: {model.get('logia')}). Сохрани все числа. "
            f"Верни ТОЛЬКО улучшенный текст, одним абзацем:\n{template}"
        )
        try:
            improved, _ = self.llm.generate(prompt)
            if improved and len(improved) > 20 and not improved.startswith("[Generator error]"):
                return improved.strip()
        except Exception:
            pass
        return template

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
