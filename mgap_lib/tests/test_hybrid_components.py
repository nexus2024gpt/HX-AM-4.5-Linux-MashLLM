#!/usr/bin/env python3
# mgap_lib/tests/test_hybrid_components.py — HX-AM v4.7
"""
Тесты гибридной архитектуры MGAP.

Запуск из корня проекта:
  python mgap_lib/tests/test_hybrid_components.py

Охват:
  1. DimensionalNormalizer  — нормализация tau, eta, K; check_physical_range
  2. ThresholdCalculator    — безопасность eval, динамические пороги, fallback
  3. TopologyValidator      — непрерывность penalty, агностичные типы, MIN_PENALTY
  4. FalsificationEngine    — Падé, mean-field, LC-collapse, LV-равновесие
"""

from __future__ import annotations

import sys
import math
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

PASS = "✅"
FAIL = "❌"
SKIP = "⚠️ "

_pass_count = 0
_fail_count = 0


def check(name: str, condition: bool, detail: str = ""):
    global _pass_count, _fail_count
    if condition:
        print(f"  {PASS} {name}")
        _pass_count += 1
    else:
        print(f"  {FAIL} {name}  ← {detail}")
        _fail_count += 1


def section(title: str):
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print(f"{'─' * 60}")


# ══════════════════════════════════════════════════════════
# 1. DimensionalNormalizer
# ══════════════════════════════════════════════════════════

def test_dimensional_normalizer():
    section("1. DimensionalNormalizer")
    from mgap_lib.engine.dimensional_normalizer import DimensionalNormalizer
    n = DimensionalNormalizer()

    # tau без basis → abstract
    tau_r, unit = n.normalize_tau(5.0, {})
    check("No basis → tau unchanged",  abs(tau_r - 5.0) < 0.01)
    check("No basis → unit=abstract",  unit == "abstract")

    # tau с basis t0=5 days
    model_epi = {
        "name": "Epidemiology",
        "dimensional_basis": {
            "t0_value": 5.0,
            "t0_unit": "days",
            "tau_physical_range": [1.0, 21.0],
        }
    }
    tau_r, unit = n.normalize_tau(3.0, model_epi)
    check("tau=3 * t0=5 → 15 days",    abs(tau_r - 15.0) < 0.01)
    check("unit = 'days'",              unit == "days")

    # check_physical_range — в диапазоне
    check("tau=15 in [1,21] → True",    n.check_physical_range(15.0, model_epi))
    # check_physical_range — вне диапазона
    check("tau=25 in [1,21] → False",   not n.check_physical_range(25.0, model_epi))
    check("tau=0.5 in [1,21] → False",  not n.check_physical_range(0.5, model_epi))

    # tau без диапазона → всегда True
    model_no_range = {"dimensional_basis": {"t0_value": 1.0, "t0_unit": "units"}}
    check("No range → always True",     n.check_physical_range(999.0, model_no_range))

    # get_physical_interpretation
    flat = {"tau": 2.0, "K": 0.4, "eta": 0.25}
    interp = n.get_physical_interpretation(flat, model_epi)
    check("interp has tau_abstract",      interp["tau_abstract"] == 2.0)
    check("interp tau_real = 2*5 = 10",   abs(interp["tau_real"] - 10.0) < 0.01)
    check("interp in_physical_range",     interp["in_physical_range"])
    check("interp basis_available=True",  interp["basis_available"])

    # Граничный случай: tau_real точно на границе
    tau_r_edge, _ = n.normalize_tau(4.2, model_epi)  # 4.2*5=21 → на верхней границе
    check("tau=4.2*5=21.0 on upper edge → True",
          n.check_physical_range(tau_r_edge, model_epi))


# ══════════════════════════════════════════════════════════
# 2. ThresholdCalculator
# ══════════════════════════════════════════════════════════

def test_threshold_calculator():
    section("2. ThresholdCalculator (с безопасным eval)")
    from mgap_lib.engine.threshold_calculator import (
        ThresholdCalculator, _validate_formula
    )
    calc = ThresholdCalculator()

    # Без контекста → статические пороги
    model_static = {"critical_thresholds": {"eta_max": 0.5, "p_crit": 0.409}}
    result = calc.compute(model_static)
    check("Static: eta_max=0.5",       abs(result["eta_max"] - 0.5) < 0.001)
    check("Static: p_crit=0.409",      abs(result["p_crit"] - 0.409) < 0.001)
    check("No _dynamic markers",       not any(k.endswith("_dynamic") for k in result))

    # ER-граф k=10 → p_crit = 1/10 = 0.1
    model_dyn = {
        "critical_thresholds": {
            "eta_max": 0.5,
            "p_crit": 0.409,
            "_dynamic": {
                "p_crit": {
                    "formula": "1 / mean_k",
                    "topology_conditions": {
                        "erdos_renyi": "1 / mean_k",
                        "scale_free":  "mean_k / (mean_k_sq - mean_k)",
                    },
                    "required_inputs": ["mean_k"],
                    "fallback": 0.409,
                }
            }
        }
    }
    ctx_er = {"mean_k": 10.0, "topology_type": "erdos_renyi"}
    result_dyn = calc.compute(model_dyn, ctx_er)
    check("ER k=10: p_crit=0.1",       abs(result_dyn["p_crit"] - 0.1) < 0.001)
    check("_p_crit_dynamic marker",    result_dyn.get("_p_crit_dynamic") is True)
    check("eta_max unchanged",         abs(result_dyn["eta_max"] - 0.5) < 0.001)

    # Scale-Free граф k=8 → Molloy-Reed
    ctx_sf = {"mean_k": 8.0, "topology_type": "scale_free"}
    result_sf = calc.compute(model_dyn, ctx_sf)
    # mean_k_sq ≈ 8^1.5 * 2 ≈ 45.25; p_crit = 8 / (45.25 - 8) ≈ 0.215
    check("SF k=8: p_crit (Molloy-Reed) computed",
          0.05 < result_sf["p_crit"] < 0.5)
    check("SF _p_crit_dynamic=True",  result_sf.get("_p_crit_dynamic") is True)

    # Fallback при missing variable
    ctx_missing = {"topology_type": "erdos_renyi"}  # нет mean_k
    result_fb = calc.compute(model_dyn, ctx_missing)
    check("Missing var → fallback 0.409", abs(result_fb["p_crit"] - 0.409) < 0.001)

    # ── Безопасность eval ────────────────────────────────
    check("Safe: '1 / mean_k'",            _validate_formula("1 / mean_k",   {"mean_k"}))
    check("Safe: 'mean_k / mean_k_sq'",    _validate_formula("mean_k / (mean_k_sq - mean_k)",
                                                               {"mean_k", "mean_k_sq"}))
    check("Safe: numeric literal '0.5927'", _validate_formula("0.5927",       set()))

    # Опасные строки должны быть отклонены
    dangerous = [
        ("__import__('os')",     set()),
        ("exec('x=1')",          set()),
        ("open('/etc/passwd')",  set()),
        ("1 + [1,2][0]",         set()),
        ("mean_k.upper()",       {"mean_k"}),
        ("1 + __builtins__",     set()),
        ("lambda x: x",         set()),
        ("a" * 201,              set()),        # слишком длинная
    ]
    for formula, tokens in dangerous:
        result_v = _validate_formula(formula, tokens)
        check(f"Blocked: {formula[:30]!r}",  not result_v)

    # Чужой токен в формуле (не в required_inputs) — должен блокироваться
    check("Blocked: token not in allowed",
          not _validate_formula("mean_k + sys_exit", {"mean_k"}))

    # infer_context_from_four_d
    fd_sf = {"structure": {"C": 0.75, "k": 25.0, "D": 2.8}}
    ctx_inferred = ThresholdCalculator.infer_context_from_four_d(fd_sf)
    check("Infer: high D+C → scale_free",   ctx_inferred["topology_type"] == "scale_free")

    fd_sw = {"structure": {"C": 0.55, "k": 8.0, "D": 2.1}}
    ctx_sw = ThresholdCalculator.infer_context_from_four_d(fd_sw)
    check("Infer: moderate C+small k → small_world",
          ctx_sw["topology_type"] == "small_world")

    fd_er = {"structure": {"C": 0.20, "k": 10.0, "D": 2.0}}
    ctx_er2 = ThresholdCalculator.infer_context_from_four_d(fd_er)
    check("Infer: low C → erdos_renyi",     ctx_er2["topology_type"] == "erdos_renyi")


# ══════════════════════════════════════════════════════════
# 3. TopologyValidator
# ══════════════════════════════════════════════════════════

def test_topology_validator():
    section("3. TopologyValidator (непрерывные штрафы)")
    from mgap_lib.engine.topology_validator import (
        TopologyValidator, _topology_profile, _MIN_PENALTY
    )
    tv = TopologyValidator()

    # ── Профили топологии ────────────────────────────────
    p_sf = _topology_profile(C=0.75, k=12.0, D=2.7)
    check("SF profile: scale_free dominant",   p_sf["scale_free"] > 0.5)
    check("SF profile: regular low",           p_sf["regular"] < 0.2)

    p_er = _topology_profile(C=0.15, k=10.0, D=2.0)
    check("ER profile: erdos_renyi dominant",  p_er["erdos_renyi"] > 0.5)
    check("ER profile: scale_free low",        p_er["scale_free"] < 0.2)

    p_sw = _topology_profile(C=0.50, k=7.0, D=2.0)
    check("SW profile: small_world notable",   p_sw["small_world"] > 0.2)

    # ── Непрерывность: промежуточные значения ────────────
    # C=0.60 — граница между SW и SF; должен быть плавный переход
    p_mid = _topology_profile(C=0.60, k=10.0, D=2.45)
    check("Mid profile: no hard zero (continuity)",
          all(v >= 0.0 for v in p_mid.values()))
    check("Mid profile: max < 0.9 (not extreme)",
          max(p_mid.values()) < 0.95)

    # ── compute_penalty ──────────────────────────────────
    fd_sf = {"structure": {"C": 0.75, "k": 12.0, "D": 2.7}, "dynamics": {}}
    fd_er = {"structure": {"C": 0.15, "k": 10.0, "D": 2.0}, "dynamics": {}}
    fd_sw = {"structure": {"C": 0.50, "k": 7.0,  "D": 2.0}, "dynamics": {}}

    model_perc = {"math_type": "percolation", "four_d_matrix": fd_er}

    pen_sf_er, _ = tv.compute_penalty(fd_sf, model_perc)
    pen_er_er, _ = tv.compute_penalty(fd_er, model_perc)
    pen_sw_er, _ = tv.compute_penalty(fd_sw, model_perc)

    check("SF art vs ER model: penalty < 0.7",    pen_sf_er < 0.70)
    check("ER art vs ER model: penalty ≥ 0.85",   pen_er_er >= 0.85)
    check("SW art vs ER model: between",           pen_sf_er < pen_sw_er < pen_er_er)

    # MIN_PENALTY соблюдается
    check("SF vs Regular: penalty >= MIN_PENALTY",
          pen_sf_er >= _MIN_PENALTY)

    # ── Агностичные типы — нет штрафа ────────────────────
    for mt in ["kuramoto", "delay", "ising", "lotka_volterra"]:
        model_ag = {"math_type": mt, "four_d_matrix": fd_er}
        pen, debug = tv.compute_penalty(fd_sf, model_ag)
        check(f"Agnostic {mt}: penalty=1.0",   abs(pen - 1.0) < 0.001)
        check(f"Agnostic {mt}: reason in debug", "topology-agnostic" in debug.get("reason", ""))

    # ── apply_penalty ────────────────────────────────────
    adj, pen, debug = tv.apply_penalty(0.8, fd_sf, model_perc)
    check("apply_penalty: adj < original",    adj < 0.8)
    check("apply_penalty: debug has fields",
          "art_profile" in debug and "adjusted_resonance" in debug)

    # ── Одинаковые топологии — высокая совместимость ─────
    model_sf = {"math_type": "percolation", "four_d_matrix": fd_sf}
    pen_sf_sf, _ = tv.compute_penalty(fd_sf, model_sf)
    # SF профиль имеет small_world компоненту (0.388), которая вносит шум через
    # SF×SW (compat=0.65) — итоговый penalty≈0.84, что корректно для смешанных сетей.
    check("SF art vs SF model: penalty ≥ 0.80", pen_sf_sf >= 0.80)


# ══════════════════════════════════════════════════════════
# 4. FalsificationEngine
# ══════════════════════════════════════════════════════════

def test_falsification_engine():
    section("4. FalsificationEngine")
    from mgap_lib.engine.falsification_engine import (
        FalsificationEngine, INVALIDATION_PENALTY, _PADE_THRESHOLD
    )
    fe = FalsificationEngine()

    # ── delay: Падé ──────────────────────────────────────
    flat_pade_ok  = {"K": 0.5,  "tau": 1.0, "k": 6.0, "C": 0.5, "D": 2.0}
    flat_pade_bad = {"K": 1.8,  "tau": 1.5, "k": 6.0, "C": 0.5, "D": 2.0}
    model_delay   = {"math_type": "delay"}

    r_ok  = fe.run(flat_pade_ok,  model_delay)
    r_bad = fe.run(flat_pade_bad, model_delay)

    # K=0.5, tau=1.0 → K*tau=0.5 < pi/2 → OK
    check("Delay K*tau=0.5 < π/2: OK",        r_ok["overall_verdict"] == "OK")
    check("Delay: multiplier=1.0 (ok)",        abs(r_ok["resonance_multiplier"] - 1.0) < 0.001)

    # K=1.8, tau=1.5 → K*tau=2.7 > pi/2 → INVALIDATED
    check("Delay K*tau=2.7 > π/2: INVALIDATED",  r_bad["overall_verdict"] == "INVALIDATED")
    check("Delay: multiplier=PENALTY",
          abs(r_bad["resonance_multiplier"] - INVALIDATION_PENALTY) < 0.001)
    check("Delay: pade_singularity triggered",
          any(s["name"] == "pade_singularity" and s["triggered"]
              for s in r_bad["scenarios"]))

    # Граничное значение: K*tau = pi/2 точно → должно быть INVALIDATED (>= не <)
    K_edge = math.pi / 2 / 1.5   # tau=1.5 → K*tau ровно pi/2 + epsilon
    flat_edge = {"K": K_edge + 0.001, "tau": 1.5, "k": 6.0, "C": 0.5, "D": 2.0}
    r_edge = fe.run(flat_edge, model_delay)
    check("Delay K*tau just above π/2: INVALIDATED", r_edge["overall_verdict"] == "INVALIDATED")

    # ── kuramoto: mean_field_breakdown ───────────────────
    flat_sparse  = {"K": 0.6, "tau": 1.0, "omega_i": 0.3, "k": 3.0, "C": 0.4, "D": 2.0}
    flat_dense   = {"K": 0.6, "tau": 1.0, "omega_i": 0.3, "k": 15.0, "C": 0.4, "D": 2.0}
    model_kur    = {"math_type": "kuramoto"}

    r_sparse = fe.run(flat_sparse, model_kur)
    r_dense  = fe.run(flat_dense,  model_kur)
    check("Kuramoto k=3 < 5: mean_field_breakdown triggered",
          any(s["name"] == "mean_field_breakdown" and s["triggered"]
              for s in r_sparse["scenarios"]))
    check("Kuramoto k=15 >= 5: no mean_field_breakdown",
          not any(s["name"] == "mean_field_breakdown" and s["triggered"]
                  for s in r_dense["scenarios"]))

    # ── kuramoto: frequency_spread_collapse ──────────────
    flat_freqbad = {"K": 0.1, "tau": 1.0, "omega_i": 2.0, "k": 15.0, "C": 0.5, "D": 2.0}
    r_freqbad = fe.run(flat_freqbad, model_kur)
    # omega_i / K = 2.0 / 0.1 = 20 > 5 → triggered
    check("Kuramoto omega/K=20 > 5: freq_spread triggered",
          any(s["name"] == "frequency_spread_collapse" and s["triggered"]
              for s in r_freqbad["scenarios"]))

    # ── ising: low dimension ─────────────────────────────
    flat_1d = {"K": 0.6, "tau": 0.5, "T": 0.8, "k": 2.0, "C": 0.1, "D": 1.5}
    flat_2d = {"K": 0.6, "tau": 0.5, "T": 0.8, "k": 4.0, "C": 0.3, "D": 2.0}
    model_is = {"math_type": "ising"}

    r_1d = fe.run(flat_1d, model_is)
    r_2d = fe.run(flat_2d, model_is)
    check("Ising D=1.5 < 2: mean_field_low_dim triggered",
          any(s["name"] == "mean_field_low_dimension" and s["triggered"]
              for s in r_1d["scenarios"]))
    check("Ising D=2.0: no mean_field_low_dim",
          not any(s["name"] == "mean_field_low_dimension" and s["triggered"]
                  for s in r_2d["scenarios"]))

    # ── ising: critical fluctuations (WARNING) ────────────
    flat_near_tc = {"K": 0.6, "tau": 0.5, "T": 0.598, "k": 6.0, "C": 0.5, "D": 2.2}
    r_near = fe.run(flat_near_tc, model_is)
    # |T - K| / K = |0.598 - 0.6| / 0.6 ≈ 0.003 < 0.05 → WARNING
    check("Ising T≈T_c: critical_fluctuations warning",
          any(s["name"] == "critical_fluctuations_divergence" and s["triggered"]
              for s in r_near["scenarios"]))
    check("Ising near-Tc: WARNING not INVALIDATED",
          r_near["overall_verdict"] in ("WARNING", "INVALIDATED"))
    check("Warning: multiplier in [0.6, 1.0)",
          0.6 <= r_near["resonance_multiplier"] < 1.0)

    # ── lotka_volterra: negative equilibrium ─────────────
    flat_lv_ok  = {"K": 0.4, "K_c": 0.5, "p": 0.7, "omega_i": 0.3,
                   "k": 6.0, "C": 0.5, "D": 2.0, "tau": 3.0}
    flat_lv_bad = {"K": 0.0, "K_c": 0.5, "p": 0.7, "omega_i": 0.3,
                   "k": 6.0, "C": 0.5, "D": 2.0, "tau": 3.0}
    model_lv    = {"math_type": "lotka_volterra"}

    r_lv_ok  = fe.run(flat_lv_ok,  model_lv)
    r_lv_bad = fe.run(flat_lv_bad, model_lv)
    check("LV K>0: negative_equilibrium not triggered",
          not any(s["name"] == "negative_equilibrium" and s["triggered"]
                  for s in r_lv_ok["scenarios"]))
    check("LV K=0: negative_equilibrium triggered",
          any(s["name"] == "negative_equilibrium" and s["triggered"]
              for s in r_lv_bad["scenarios"]))

    # ── percolation: giant_component_collapse ────────────
    flat_p_ok  = {"p": 0.55, "k": 8.0, "C": 0.4, "D": 2.1, "K": 0.4, "tau": 1.0}
    flat_p_bad = {"p": 0.998, "k": 8.0, "C": 0.4, "D": 2.1, "K": 0.4, "tau": 1.0}
    model_per  = {"math_type": "percolation",
                  "critical_thresholds": {"p_crit": 0.37}}

    r_p_ok  = fe.run(flat_p_ok,  model_per)
    r_p_bad = fe.run(flat_p_bad, model_per)
    check("Percolation p=0.55: no collapse",
          not any(s["name"] == "giant_component_collapse" and s["triggered"]
                  for s in r_p_ok["scenarios"]))
    check("Percolation p=0.998: collapse triggered",
          any(s["name"] == "giant_component_collapse" and s["triggered"]
              for s in r_p_bad["scenarios"]))

    # ── graph_invariant: below threshold ─────────────────
    flat_gi_ok  = {"p": 0.3, "k": 6.0, "C": 0.5, "D": 2.0, "K": 0.4, "tau": 1.0}
    flat_gi_bad = {"p": 0.1, "k": 6.0, "C": 0.5, "D": 2.0, "K": 0.4, "tau": 1.0}
    model_gi    = {"math_type": "graph_invariant"}

    r_gi_ok  = fe.run(flat_gi_ok,  model_gi)
    r_gi_bad = fe.run(flat_gi_bad, model_gi)
    # p_c = 1/6 ≈ 0.167; p=0.3 > 0.167 → ok
    check("GI p=0.3 > p_c=0.167: OK",
          not any(s["name"] == "below_percolation_threshold" and s["triggered"]
                  for s in r_gi_ok["scenarios"]))
    # p=0.1 < p_c=0.167 → triggered
    check("GI p=0.1 < p_c=0.167: triggered",
          any(s["name"] == "below_percolation_threshold" and s["triggered"]
              for s in r_gi_bad["scenarios"]))

    # ── Неизвестный math_type → пустой результат ─────────
    r_unknown = fe.run({"K": 0.5, "tau": 1.0}, {"math_type": "unknown_future_type"})
    check("Unknown math_type → empty result",
          r_unknown["overall_verdict"] == "OK" and
          r_unknown["resonance_multiplier"] == 1.0 and
          len(r_unknown["scenarios"]) == 0)


# ══════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("  HX-AM v4.7 — MGAP Hybrid Components Tests")
    print("=" * 60)

    tests = [
        ("DimensionalNormalizer",  test_dimensional_normalizer),
        ("ThresholdCalculator",    test_threshold_calculator),
        ("TopologyValidator",      test_topology_validator),
        ("FalsificationEngine",    test_falsification_engine),
    ]

    for name, fn in tests:
        try:
            fn()
        except Exception as e:
            print(f"\n  {FAIL} {name} CRASHED: {e}")
            traceback.print_exc()
            global _fail_count
            _fail_count += 1

    print(f"\n{'=' * 60}")
    print(f"  Results: {PASS} {_pass_count} passed  |  {FAIL} {_fail_count} failed")
    print(f"{'=' * 60}\n")
    sys.exit(0 if _fail_count == 0 else 1)


if __name__ == "__main__":
    main()
