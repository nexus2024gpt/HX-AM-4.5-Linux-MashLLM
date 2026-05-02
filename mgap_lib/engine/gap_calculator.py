# mgap_lib/engine/gap_calculator.py — MGAP Library v1.0
"""
GapCalculator — вычисляет метрический разрыв (Gap) между параметрами артефакта
и критическими порогами отраслевой модели.

Gap = насколько параметры артефакта ПРЕВЫШАЮТ допустимые пороги.
Если параметр в норме — gap=0. Если превышен — gap > 0 (доля превышения).

Компоненты:
  eta_gap  = max(0, (artifact_eta - model_eta_max) / model_eta_max)
  tau_gap  = max(0, (artifact_tau - model_tau_max) / model_tau_max)
  K_gap    = max(0, (model_K_min - artifact_K)   / model_K_min)  ← K слишком мал

Composite gap:
  mode="max"  → max(eta_gap, tau_gap, K_gap)      — консервативный
  mode="mean" → среднее арифметическое            — сглаженный
  mode="rms"  → RMS                               — квадратичный

Risk levels (по composite gap):
  none:     gap = 0
  monitor:  0 < gap ≤ 0.20   (+20% превышение)
  moderate: 0.20 < gap ≤ 0.50
  critical: gap > 0.50
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class GapComponents:
    eta_gap:   float = 0.0
    tau_gap:   float = 0.0
    K_gap:     float = 0.0
    composite: float = 0.0
    risk_level: str  = "none"   # none | monitor | moderate | critical
    mode:      str   = "max"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "eta_gap":    round(self.eta_gap, 4),
            "tau_gap":    round(self.tau_gap, 4),
            "K_gap":      round(self.K_gap, 4),
            "composite":  round(self.composite, 4),
            "risk_level": self.risk_level,
            "mode":       self.mode,
        }

    @property
    def is_warning(self) -> bool:
        return self.risk_level in ("moderate", "critical")


class GapCalculator:
    """
    Вычисляет метрические разрывы между параметрами артефакта и моделью.

    Пример:
        calc = GapCalculator()
        artifact_params = {"eta": 0.45, "tau": 5.5, "K": 0.2}
        model_thresholds = {"eta_max": 0.35, "tau_max": 4.5, "K_min": 0.3}
        gap = calc.compute(artifact_params, model_thresholds)
        print(gap.risk_level)   # "moderate"
        print(gap.composite)    # 0.286
    """

    RISK_THRESHOLDS = {
        "none":     0.0,
        "monitor":  0.20,
        "moderate": 0.50,
    }

    def compute(
        self,
        artifact_params: Dict[str, float],
        model_thresholds: Dict[str, float],
        mode: str = "max",
    ) -> GapComponents:
        """
        Вычисляет gap.

        Args:
            artifact_params:  {"eta": ..., "tau": ..., "K": ...}
            model_thresholds: {"eta_max": ..., "tau_max": ..., "K_min": ...}
            mode:             "max" | "mean" | "rms"
        """
        eta_a  = float(artifact_params.get("eta", 0.0))
        tau_a  = float(artifact_params.get("tau", 0.0))
        K_a    = float(artifact_params.get("K",   0.0))

        eta_max = float(model_thresholds.get("eta_max", float("inf")))
        tau_max = float(model_thresholds.get("tau_max", float("inf")))
        K_min   = float(model_thresholds.get("K_min",  0.0))

        eta_gap = max(0.0, (eta_a - eta_max) / max(eta_max, 1e-9)) if math.isfinite(eta_max) else 0.0
        tau_gap = max(0.0, (tau_a - tau_max) / max(tau_max, 1e-9)) if math.isfinite(tau_max) else 0.0
        K_gap   = max(0.0, (K_min - K_a)    / max(K_min, 1e-9))    if K_min > 0 else 0.0

        gaps = [eta_gap, tau_gap, K_gap]

        if mode == "max":
            composite = max(gaps)
        elif mode == "mean":
            composite = sum(gaps) / 3
        elif mode == "rms":
            composite = math.sqrt(sum(g**2 for g in gaps) / 3)
        else:
            composite = max(gaps)

        risk_level = self._classify_risk(composite)

        return GapComponents(
            eta_gap=round(eta_gap, 4),
            tau_gap=round(tau_gap, 4),
            K_gap=round(K_gap, 4),
            composite=round(composite, 4),
            risk_level=risk_level,
            mode=mode,
        )

    def compute_from_artifact_and_model(
        self,
        four_d: Dict,
        model_thresholds: Dict[str, float],
        mode: str = "max",
    ) -> GapComponents:
        """Удобный метод: принимает four_d_matrix артефакта напрямую."""
        dyn = four_d.get("dynamics", {})
        inf = four_d.get("influence", {})
        tim = four_d.get("time", {})
        params = {
            "eta": float(inf.get("eta", 0.2)),
            "tau": float(tim.get("tau", 0.5)),
            "K":   float(dyn.get("K",   0.35)),
        }
        return self.compute(params, model_thresholds, mode)

    def describe_risk(self, gap: GapComponents) -> str:
        """Читаемое описание риска на русском."""
        if gap.risk_level == "none":
            return "Параметры в норме. Мониторинг не требуется."
        elif gap.risk_level == "monitor":
            parts = self._gap_parts(gap)
            return f"Лёгкое превышение ({', '.join(parts)}). Рекомендуется мониторинг."
        elif gap.risk_level == "moderate":
            parts = self._gap_parts(gap)
            return f"Умеренное превышение ({', '.join(parts)}). Требуется усиление буфера."
        else:
            parts = self._gap_parts(gap)
            return f"КРИТИЧЕСКОЕ превышение ({', '.join(parts)}). Немедленно скорректировать параметры."

    @staticmethod
    def _gap_parts(gap: GapComponents) -> list:
        parts = []
        if gap.eta_gap > 0:
            parts.append(f"η +{gap.eta_gap*100:.0f}%")
        if gap.tau_gap > 0:
            parts.append(f"τ +{gap.tau_gap*100:.0f}%")
        if gap.K_gap > 0:
            parts.append(f"K -{gap.K_gap*100:.0f}%")
        return parts or ["composite={:.3f}".format(gap.composite)]

    @staticmethod
    def _classify_risk(composite: float) -> str:
        if composite <= 0.0:
            return "none"
        elif composite <= 0.20:
            return "monitor"
        elif composite <= 0.50:
            return "moderate"
        else:
            return "critical"

    def summary_table(self, artifact_params: Dict, thresholds: Dict) -> str:
        """ASCII-таблица для CLI/логов."""
        gap = self.compute(artifact_params, thresholds)
        lines = [
            "┌─────────────┬──────────┬──────────┬──────────┐",
            "│ Параметр    │ Арт-факт │  Порог   │   Gap    │",
            "├─────────────┼──────────┼──────────┼──────────┤",
            f"│ η (noise)   │ {artifact_params.get('eta', '?'):>8.3f} │ {thresholds.get('eta_max', '∞'):>8} │ {gap.eta_gap:>8.3f} │",
            f"│ τ (delay)   │ {artifact_params.get('tau', '?'):>8.3f} │ {thresholds.get('tau_max', '∞'):>8} │ {gap.tau_gap:>8.3f} │",
            f"│ K (coupling)│ {artifact_params.get('K', '?'):>8.3f} │ {thresholds.get('K_min', 0):>8} │ {gap.K_gap:>8.3f} │",
            "├─────────────┼──────────┼──────────┼──────────┤",
            f"│ Composite   │          │          │ {gap.composite:>8.3f} │",
            f"│ Risk level  │          │          │ {gap.risk_level:>8} │",
            "└─────────────┴──────────┴──────────┴──────────┘",
        ]
        return "\n".join(lines)
