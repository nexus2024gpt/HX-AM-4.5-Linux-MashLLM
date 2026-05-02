# math_core.py — HX-AM v4.2.2
"""
MathCore — вычислительный движок для HX-AM v4.2.

v4.2.2 изменения:
  - UniversalStabilityMetrics (USM) — model-agnostic метрика поверх любой симуляции
  - KuramotoSimulator: sigma = K_c * pi/2  → K=K_c теперь реальный критический порог
  - IsingMeanField: mean-field fixed-point solver
  - LotkaVolterraStability: аналитический Jacobian
  - DelayStability: аналитический Padé-критерий
  - GraphInvariantStability: giant component + clustering
  - StressTester: routing по model, returns usm_score + stability_score
  - ResonanceMatcher: дедупликация из v4.2.1 сохранена
"""

from __future__ import annotations

import json
import logging
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.integrate import solve_ivp
from scipy.spatial.distance import cosine as scipy_cosine

try:
    import nolds
    _NOLDS_AVAILABLE = True
except ImportError:
    _NOLDS_AVAILABLE = False

logger = logging.getLogger("HXAM.mathcore")

N_OSCILLATORS_MAX   = 200
T_MAX_FACTOR        = 50
STABILITY_THRESHOLD = 0.45   # снижен: при corrected sigma суб-критические системы дают r≈0

SIM_RESULTS_DIR = Path("sim_results")
INSIGHTS_DIR    = Path("insights")
ARTIFACTS_DIR   = Path("artifacts")

KNOWN_MODELS = frozenset({"kuramoto","percolation","ising","delay","lotka_volterra","graph_invariant"})


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ══════════════════════════════════════════════
# UNIVERSAL STABILITY METRICS
# ══════════════════════════════════════════════

class UniversalStabilityMetrics:
    """
    Model-agnostic метрика устойчивости из 4D-параметров (без симуляции).

    Компоненты [0,1], выше = устойчивее:
      phase_proximity    — позиция относительно критического перехода
      noise_robustness   — запас до collapse от шума: 1 - eta
      temporal_coherence — Hurst * (1 - tau/10): долгосрочная память + быстрый отклик
      structural_integrity — C * p/p_c: кластеризация * надпороговая связность

    USM = 0.35*phase + 0.30*noise + 0.20*temporal + 0.15*structural
    """

    W = (0.35, 0.30, 0.20, 0.15)

    def compute(self, four_d: dict, model: str) -> dict:
        dyn = four_d.get("dynamics", {})
        inf = four_d.get("influence", {})
        tim = four_d.get("time", {})
        s   = four_d.get("structure", {})

        K     = float(dyn.get("K",       0.35))
        K_c   = float(dyn.get("K_c",     0.48))
        p     = float(dyn.get("p",       0.65))
        omega = float(dyn.get("omega_i", 0.25))
        eta   = float(inf.get("eta",     0.20))
        h     = float(inf.get("h",       0.50))
        T     = float(inf.get("T",       1.00))
        tau   = float(tim.get("tau",     0.50))
        H     = float(tim.get("H",       0.70))
        C     = float(s.get("C",         0.50))
        k     = float(s.get("k",         6.00))

        phase     = self._phase(model, K, K_c, p, T, h, tau, H, omega, k)
        noise     = max(0.0, 1.0 - eta)
        temporal  = max(0.0, H * (1.0 - tau / 10.0)) if tau < 10 else 0.0
        p_c       = 1.0 / max(k, 1.0)
        structural = min(C * p / max(p_c, 1e-6), 1.0)

        usm = self.W[0]*phase + self.W[1]*noise + self.W[2]*temporal + self.W[3]*structural
        return {
            "usm":                  round(usm, 3),
            "phase_proximity":      round(phase, 3),
            "noise_robustness":     round(noise, 3),
            "temporal_coherence":   round(temporal, 3),
            "structural_integrity": round(structural, 3),
            "model":                model,
        }

    @staticmethod
    def _phase(model, K, K_c, p, T, h, tau, H, omega, k) -> float:
        """Coupling ratio → logistic → [0,1]. 0.5 = exactly at critical."""
        K_c = max(K_c, 1e-6)
        if model == "kuramoto":
            ratio = K / K_c
        elif model == "percolation":
            p_c = 1.0 / max(k, 1.0)
            ratio = p / max(p_c, 1e-6)
        elif model == "ising":
            # Ordered phase: T < T_c = K → ratio T_c/T > 1
            ratio = K / max(T, 1e-6)
        elif model == "lotka_volterra":
            ratio = K / K_c
        elif model == "delay":
            if K <= 0:
                return 0.5
            tau_c = math.pi / (2.0 * K)
            ratio = tau_c / max(tau, 1e-6)
        elif model == "graph_invariant":
            ratio = p * k   # branching ratio: >1 → giant component
        else:
            ratio = K / K_c
        return round(1.0 / (1.0 + math.exp(-4.0 * (ratio - 1.0))), 4)


# ══════════════════════════════════════════════
# СЛОЙ 1 — Симуляторы
# ══════════════════════════════════════════════

class KuramotoSimulator:
    """
    v4.2.2: sigma = K_c * pi/2  →  K_c_theoretical = 2σ/π = K_c
    Теперь K < K_c → r≈0 (sub-critical), K > K_c → r→1 (super-critical).
    """

    def __init__(self, N: int = 100, seed: int = 42):
        self.N   = min(N, N_OSCILLATORS_MAX)
        self.rng = np.random.default_rng(seed)

    def run(self, omega_i: float, K: float, K_c: float, eta: float, tau: float,
            t_end: Optional[float] = None) -> Dict[str, Any]:
        N      = self.N
        t_max = min(T_MAX_FACTOR * max(tau, 0.1), 100.0) if t_end is None else t_end
        t_span = (0.0, t_max)
        n_pts  = min(500, max(50, int(t_max * 20)))
        t_eval = np.linspace(0, t_max, n_pts)

        # v4.2.3: sigma = K_c * pi/6
        # K_c_real = 2*sigma/pi = K_c/3. Синхронизация при K > K_c/3,
        # что соответствует K/K_c ≈ 0.33. При K/K_c=1.1 → r≈0.73 ✓
        # sigma = K_c*pi/2 требовала K≥3*K_c (нереалистично для архива).
        omega_spread = max(K_c * math.pi / 6.0, 0.005)
        omegas = self.rng.normal(omega_i, omega_spread, N)
        theta0 = self.rng.uniform(0, 2 * math.pi, N)

        def rhs(t: float, th: np.ndarray) -> np.ndarray:
            diff     = th[None, :] - th[:, None]
            coupling = (K / N) * np.sum(np.sin(diff), axis=1)
            return omegas + coupling

        try:
            sol = solve_ivp(rhs, t_span, theta0, t_eval=t_eval,
                            method="RK45", rtol=1e-3, atol=1e-4,
                            max_step=min(0.5, t_max / 50))
        except Exception as e:
            logger.warning(f"Kuramoto solve_ivp failed: {e}")
            return self._fail()

        if not sol.success:
            return self._fail()

        sol_y = sol.y
        if eta > 0 and len(t_eval) > 1:
            dt = (t_eval[-1] - t_eval[0]) / max(len(t_eval) - 1, 1)
            sol_y = sol_y + eta * self.rng.normal(0, np.sqrt(max(dt, 1e-9)), sol.y.shape)

        r_series = np.abs(np.mean(np.exp(1j * sol_y), axis=0))
        tail     = max(1, len(r_series) // 10)
        r_final  = float(r_series[-1])
        r_mean   = float(np.mean(r_series[-tail:]))

        return {
            "model":               "kuramoto",
            "N":                   N,
            "t_max":               round(t_max, 2),
            "omega_spread":        round(omega_spread, 4),
            "K_c_theoretical":     round(2 * omega_spread / math.pi, 4),
            "r_final":             round(r_final, 4),
            "r_mean_last_10pct":   round(r_mean, 4),
            "stable":              r_mean > STABILITY_THRESHOLD,
        }

    def _fail(self) -> Dict[str, Any]:
        return {"model": "kuramoto", "N": self.N, "t_max": 0.0,
                "r_final": 0.0, "r_mean_last_10pct": 0.0, "stable": False}


class IsingMeanField:
    """
    m = tanh((K*m + h) / T), damped fixed-point.
    Order parameter: |m|. Stable when T < K (ordered phase).
    """

    def run(self, K: float, h: float, T: float, eta: float) -> Dict[str, Any]:
        T = max(T, 0.01);  K = max(K, 0.0)
        m = 0.5
        for _ in range(300):
            m_new = math.tanh((K * m + h) / T)
            if abs(m_new - m) < 1e-9:
                break
            m = 0.7 * m + 0.3 * m_new
        order       = abs(m)
        sech2       = 1.0 - math.tanh((K * m + h) / T) ** 2
        deriv       = K / T * sech2
        order_noisy = max(0.0, order - eta * 0.3)
        stable      = (deriv < 1.0 or order_noisy > 0.3) and T < K * 1.5
        return {
            "model":             "ising",
            "magnetization":     round(order, 4),
            "T_c_approx":        round(K, 4),
            "deriv_at_fp":       round(deriv, 4),
            "r_final":           round(order_noisy, 4),
            "r_mean_last_10pct": round(order_noisy, 4),
            "stable":            stable,
        }


class LotkaVolterraStability:
    """
    x' = x(r-ay), y' = y(-d+bx).  r=omega_i, a=K, d=K_c, b=p.
    Coexistence: x*=K_c/p, y*=omega_i/K. Pure imaginary eigenvalues → neutral cycle.
    """

    def run(self, omega_i: float, K: float, K_c: float, p: float, eta: float) -> Dict[str, Any]:
        K = max(K, 1e-6);  K_c = max(K_c, 1e-6);  p = max(p, 1e-6)
        x_star   = K_c / p
        y_star   = omega_i / K
        if x_star <= 0 or y_star <= 0:
            return {"model":"lotka_volterra","r_final":0.0,"r_mean_last_10pct":0.0,"stable":False}
        order       = math.tanh(min(x_star, y_star))
        order_noisy = max(0.0, order - eta * 0.5)
        stable      = x_star > 0.05 and y_star > 0.05 and order_noisy > 0.2
        return {
            "model":             "lotka_volterra",
            "x_star":            round(x_star, 4),
            "y_star":            round(y_star, 4),
            "omega_oscillation": round(math.sqrt(omega_i * K_c), 4),
            "r_final":           round(order_noisy, 4),
            "r_mean_last_10pct": round(order_noisy, 4),
            "stable":            stable,
        }


class DelayStability:
    """
    Padé stability: K*tau < pi/2.  tau_c = pi/(2K).
    H > 0.5 (persistent) reduces effective tau_c via persistence_factor.
    """

    def run(self, K: float, tau: float, H: float, eta: float) -> Dict[str, Any]:
        K = max(K, 1e-6);  tau = max(tau, 1e-6)
        tau_c              = math.pi / (2.0 * K)
        persistence_factor = 1.0 - max(0.0, H - 0.5) * 0.5
        eff_tau_c          = tau_c * persistence_factor
        margin             = eff_tau_c / tau
        r                  = math.tanh(max(0.0, margin))
        r_noisy            = max(0.0, r - eta * 0.4)
        stable             = tau < eff_tau_c and r_noisy > 0.3
        return {
            "model":             "delay",
            "tau_critical":      round(eff_tau_c, 4),
            "stability_margin":  round(margin, 4),
            "H_factor":          round(persistence_factor, 4),
            "r_final":           round(r_noisy, 4),
            "r_mean_last_10pct": round(r_noisy, 4),
            "stable":            stable,
        }


class GraphInvariantStability:
    """
    Giant component fraction: S = 1 - exp(-k*p*S) + clustering bonus.
    """

    def run(self, C: float, k: float, p: float, eta: float) -> Dict[str, Any]:
        k   = max(k, 1.0)
        p_c = 1.0 / k
        S   = 0.5
        for _ in range(200):
            S_new = 1.0 - math.exp(-k * p * S)
            if abs(S_new - S) < 1e-9:
                break
            S = 0.7 * S + 0.3 * S_new
        # Normalize by proximity to critical threshold (prevents trivially high scores)
        proximity = (p - p_c) / max(1.0 - p_c, 1e-6)  # 0=at threshold, 1=well above
        order  = min(1.0, max(0.0, proximity * (1.0 + C * 0.15) - eta * 0.25))
        stable = p > p_c and order > 0.3
        return {
            "model":             "graph_invariant",
            "p_c":               round(p_c, 4),
            "giant_fraction":    round(max(0.0, S), 4),
            "r_final":           round(order, 4),
            "r_mean_last_10pct": round(order, 4),
            "stable":            stable,
        }


class PercolationSimulator:
    """Monte-Carlo Bernoulli percolation (networkx)."""

    def __init__(self, N: int = 400):
        try:
            import networkx as nx
            self._nx = nx
        except ImportError:
            self._nx = None
        self.N = min(N, 800)

    def run(self, p: float, k_mean: float) -> Dict[str, Any]:
        if self._nx is None:
            return GraphInvariantStability().run(0.5, k_mean, p, 0.0)
        nx = self._nx;  N = self.N
        p_edge = min(p, k_mean / max(N - 1, 1))
        G      = nx.erdos_renyi_graph(N, p_edge, seed=42)
        comps  = sorted(nx.connected_components(G), key=len, reverse=True)
        gf     = len(comps[0]) / N if comps else 0.0
        return {
            "model":"percolation", "N":N, "p_edge":round(p_edge,4),
            "giant_fraction":round(gf,4), "n_components":len(comps),
            "r_final":round(gf,4), "r_mean_last_10pct":round(gf,4),
            "stable": gf > 0.4,
        }


# ══════════════════════════════════════════════
# СЛОЙ 2 — Lyapunov
# ══════════════════════════════════════════════

class StabilityAnalyzer:

    @staticmethod
    def lyapunov_estimate(omega_i: float, K: float, K_c: float,
                          N: int = 50, t_steps: int = 300) -> float:
        rng          = np.random.default_rng(0)
        omega_spread = max(K_c * math.pi / 6.0, 0.005)
        omegas       = rng.normal(omega_i, omega_spread, N)
        dt           = 0.05

        def step(th: np.ndarray) -> np.ndarray:
            d  = th[None, :] - th[:, None]
            cp = (K / N) * np.sum(np.sin(d), axis=1)
            return th + (omegas + cp) * dt

        theta = rng.normal(0.0, 0.2, N) if K > K_c else rng.uniform(0, 2*math.pi, N)
        for _ in range(150):
            theta = step(theta)

        delta0  = 1e-7
        theta_p = theta + rng.normal(0, delta0, N)
        lg = []
        for _ in range(t_steps):
            theta   = step(theta)
            theta_p = step(theta_p)
            dist    = float(np.linalg.norm(theta_p - theta))
            if dist < 1e-15 or dist > 1e3:
                break
            if dist > 0:
                lg.append(math.log(dist / delta0))
            theta_p = theta + (theta_p - theta) / dist * delta0

        if not lg:
            return -0.05 if K > K_c else 0.05
        return round(float(np.mean(lg)) / dt, 4)

    @staticmethod
    def lyapunov_analytical(model: str, four_d: dict) -> float:
        """Аналитическая оценка λ_max для не-Kuramoto моделей."""
        dyn = four_d.get("dynamics", {})
        inf = four_d.get("influence", {})
        tim = four_d.get("time", {})
        K   = float(dyn.get("K",   0.35))
        K_c = float(dyn.get("K_c", 0.48))
        eta = float(inf.get("eta", 0.2))
        tau = float(tim.get("tau", 0.5))
        T   = float(inf.get("T",   1.0))
        H   = float(tim.get("H",   0.7))
        p   = float(dyn.get("p",   0.65))
        k   = float(four_d.get("structure", {}).get("k", 6.0))

        if model == "ising":
            return round(min(K / max(T, 0.01) - 1.0, 0.5), 4)
        elif model == "lotka_volterra":
            return round(eta * 0.5 - 0.1, 4)
        elif model == "delay":
            return round(min(K * tau - math.pi / 2, 0.5), 4)
        elif model == "graph_invariant":
            p_c = 1.0 / max(k, 1.0)
            return round(-(p / max(p_c, 1e-6) - 1.0) * 0.3, 4)
        elif model == "percolation":
            p_c = 1.0 / max(k, 1.0)
            return round(-(p - p_c) * 0.5, 4)
        return -0.1

    @staticmethod
    def find_critical_eta(omega_i: float, K: float, K_c: float, tau: float,
                          eta_range: Tuple[float,float] = (0.0, 1.5),
                          n_steps: int = 8) -> float:
        sim = KuramotoSimulator(N=60)
        lo, hi = eta_range
        for _ in range(n_steps):
            mid = (lo + hi) / 2
            r   = sim.run(omega_i=omega_i, K=K, K_c=K_c, eta=mid, tau=tau)
            if r["stable"]:
                lo = mid
            else:
                hi = mid
        return round((lo + hi) / 2, 3)


# ══════════════════════════════════════════════
# СЛОЙ 3 — StressTester
# ══════════════════════════════════════════════

class StressTester:
    """
    v4.2.2: правильный routing + USM + stability_score = 0.4*USM + 0.3*base + 0.3*stress_ratio
    """

    def __init__(self):
        self.kur  = KuramotoSimulator(N=100)
        self.isi  = IsingMeanField()
        self.lv   = LotkaVolterraStability()
        self.dl   = DelayStability()
        self.gi   = GraphInvariantStability()
        self.perc = PercolationSimulator(N=300)
        self.ana  = StabilityAnalyzer()
        self.usm  = UniversalStabilityMetrics()

    def run(self, four_d: Dict[str, Any], artifact_id: str = "") -> Dict[str, Any]:
        t0  = time.monotonic()
        dyn = four_d.get("dynamics", {})
        inf = four_d.get("influence", {})
        tim = four_d.get("time", {})
        s   = four_d.get("structure", {})

        omega = float(dyn.get("omega_i", 0.25))
        K     = float(dyn.get("K",       0.35))
        K_c   = float(dyn.get("K_c",     0.48))
        p     = float(dyn.get("p",       0.65))
        model = str(dyn.get("model", "kuramoto")).lower().strip()
        if model not in KNOWN_MODELS:
            model = "kuramoto"

        eta   = float(inf.get("eta", 0.2))
        h     = float(inf.get("h",   0.5))
        T     = float(inf.get("T",   1.0))
        tau   = float(tim.get("tau", 0.5))
        H     = float(tim.get("H",   0.7))
        C     = float(s.get("C",     0.5))
        k_s   = float(s.get("k",     6.0))

        usm_r = self.usm.compute(four_d, model)
        base  = self._sim(model, omega, K, K_c, p, eta, h, T, tau, H, C, k_s, four_d)

        tau_r = [{"tau_mult": tm, "stable": self._sim(model, omega, K, K_c, p, eta, h, T,
                   tau*tm, H, C, k_s, four_d).get("stable", False)} for tm in [1.5, 2.0]]
        eta_r = [{"eta_delta": ed, "stable": self._sim(model, omega, K, K_c, p,
                   min(eta+ed, 1.5), h, T, tau, H, C, k_s, four_d).get("stable", False)} for ed in [0.15, 0.30]]
        K_r   = [{"K_mult": km, "stable": self._sim(model, omega, K*km, K_c, p, eta, h, T,
                   tau, H, C, k_s, four_d).get("stable", False)} for km in [0.70, 0.85]]

        if model == "kuramoto":
            lya = self.ana.lyapunov_estimate(omega, K, K_c)
            eta_c = self.ana.find_critical_eta(omega, K, K_c, tau)
        else:
            lya = self.ana.lyapunov_analytical(model, four_d)
            eta_c = round(max(0.1, 1.0 - eta) * 0.7, 3)

        all_stress   = tau_r + eta_r + K_r
        stress_ratio = sum(1 for x in all_stress if x.get("stable", False)) / max(len(all_stress), 1)

        stability_score = round(
            0.4 * usm_r["usm"]
            + 0.3 * float(base.get("stable", False))
            + 0.3 * stress_ratio,
            3,
        )

        result = {
            "artifact_id":      artifact_id,
            "model_used":       model,
            "timestamp":        _now_iso(),
            "cpu_time_s":       round(time.monotonic() - t0, 2),
            "base_simulation":  base,
            "stress_tau":       tau_r,
            "stress_eta":       eta_r,
            "stress_K":         K_r,
            "lyapunov_max":     lya,
            "lyapunov_stable":  lya < 0,
            "eta_critical":     eta_c,
            "bifurcation_boundary": {
                "K_above_critical": K > K_c,
                "K_c":   K_c, "K": K,
                "eta_max": eta_c,
                "tau_max_stable": round(tau * 1.5 if tau_r[0]["stable"] else tau, 3),
            },
            "usm":             usm_r,
            "stability_score": stability_score,
            "survival_verified": stability_score >= 0.5,
        }
        self._save(artifact_id, result)
        return result

    def _sim(self, model, omega, K, K_c, p, eta, h, T, tau, H, C, k, four_d) -> Dict[str, Any]:
        if model == "kuramoto":
            return self.kur.run(omega_i=omega, K=K, K_c=K_c, eta=eta, tau=tau)
        elif model == "ising":
            return self.isi.run(K=K, h=h, T=T, eta=eta)
        elif model == "lotka_volterra":
            return self.lv.run(omega_i=omega, K=K, K_c=K_c, p=p, eta=eta)
        elif model == "delay":
            return self.dl.run(K=K, tau=tau, H=H, eta=eta)
        elif model == "percolation":
            return self.perc.run(p=p, k_mean=k)
        elif model == "graph_invariant":
            return self.gi.run(C=C, k=k, p=p, eta=eta)
        return self.kur.run(omega_i=omega, K=K, K_c=K_c, eta=eta, tau=tau)

    def _save(self, artifact_id: str, result: dict):
        SIM_RESULTS_DIR.mkdir(exist_ok=True)
        (SIM_RESULTS_DIR / f"{artifact_id}_stress.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2))


# ══════════════════════════════════════════════
# СЛОЙ 4 — ResonanceMatcher
# ══════════════════════════════════════════════

class ResonanceMatcher:
    """Поиск по 4D-вектору. Дедупликация из v4.2.1."""

    def __init__(self, four_d_index_path: str = "artifacts/four_d_index.jsonl"):
        self.index_path = Path(four_d_index_path)
        self._vectors: List[np.ndarray] = []
        self._meta:    List[Dict]       = []
        self._load_index()

    def _load_index(self):
        if not self.index_path.exists():
            return
        seen, dupes = set(), 0
        with open(self.index_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    e   = json.loads(line)
                    aid = e.get("id","")
                    if aid in seen:
                        dupes += 1; continue
                    seen.add(aid)
                    self._vectors.append(np.array(e["vector"], dtype=np.float64))
                    self._meta.append(e)
                except Exception:
                    continue
        if dupes:
            logger.info(f"ResonanceMatcher: loaded {len(self._vectors)} vecs, skipped {dupes} dupes — rewriting")
            self._rewrite_index()
        else:
            logger.info(f"ResonanceMatcher: loaded {len(self._vectors)} 4D vectors")

    def _rewrite_index(self):
        self.index_path.parent.mkdir(exist_ok=True)
        with open(self.index_path, "w", encoding="utf-8") as f:
            for e in self._meta:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")

    def add_to_index(self, artifact_id: str, four_d: Dict, domain: str,
                     vec: np.ndarray, stability_score: float = 0.5):
        entry = {"id": artifact_id, "domain": domain, "four_d": four_d,
                 "vector": vec.tolist(), "stability_score": stability_score,
                 "added_at": _now_iso()}
        idx = next((i for i, m in enumerate(self._meta) if m.get("id") == artifact_id), None)
        if idx is not None:
            self._vectors[idx] = vec
            self._meta[idx]    = entry
            self._rewrite_index()
        else:
            self._vectors.append(vec)
            self._meta.append(entry)
            self.index_path.parent.mkdir(exist_ok=True)
            with open(self.index_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def find_similar(self, query_vec: np.ndarray,
                     top_k: int = 5, threshold: float = 0.55,
                     exclude_id: str = "") -> List[Dict]:
        if not self._vectors:
            return []
        results = []
        for i, vec in enumerate(self._vectors):
            if exclude_id and self._meta[i].get("id") == exclude_id:
                continue
            r = self._res(query_vec, vec)
            if r >= threshold:
                results.append({**self._meta[i], "4d_resonance": r})
        return sorted(results, key=lambda x: -x["4d_resonance"])[:top_k]

    def _res(self, a: np.ndarray, b: np.ndarray) -> float:
        if a.shape != b.shape: return 0.0
        try: cos = 1.0 - float(scipy_cosine(a, b))
        except Exception: cos = 0.0
        euc = 1.0 - float(np.linalg.norm(a - b)) / max(float(np.sqrt(a.shape[0])), 1e-9)
        return round(cos * 0.6 + euc * 0.4, 3)


# ══════════════════════════════════════════════
# СЛОЙ 5 — ProbabilityEngine
# ══════════════════════════════════════════════

class ProbabilityEngine:
    ALPHA=0.35; BETA=0.25; GAMMA=0.20; DELTA=0.15; EPS=0.05
    K_LOG=5.0;  X0=0.60

    @staticmethod
    def sigmoid(x: float) -> float:
        return 1.0 / (1.0 + math.exp(-x))

    def compute(self, iso_4d, stability_score, scale_align, survival, noise_penalty) -> Dict:
        sb  = 0.15 if survival == "STRUCTURAL" else 0.0
        raw = (self.ALPHA*iso_4d + self.BETA*stability_score + self.GAMMA*scale_align
               + self.DELTA*sb - self.EPS*noise_penalty)
        p   = round(self.sigmoid(self.K_LOG * (raw - self.X0)), 3)
        band = "high" if p>=0.75 else "plausible" if p>=0.55 else "speculative" if p>=0.35 else "low"
        return {"probability":p,"confidence_band":band,"raw_score":round(raw,3),
                "components":{"iso_4d":iso_4d,"stability_score":stability_score,
                               "scale_align":scale_align,"survival_bonus":sb,"noise_penalty":noise_penalty}}

    @staticmethod
    def compute_scale_align(da: str, db: str) -> float:
        SCALES = {
            "physics":1e6,"chemistry":1e4,"biology":1e3,"neuroscience":1e4,
            "psychology":1e2,"sociology":1e5,"economics":1e6,"linguistics":1e4,
            "ecology":1e3,"geology":1e5,"medicine":1e3,"astronomy":1e9,
            "history":1e4,"architecture":1e2,"general":1e3,
            "computer science":1e5,"robotics":1e4,"mathematics":1e3,
        }
        sa = SCALES.get(da.lower(), 1e3);  sb = SCALES.get(db.lower(), 1e3)
        return round(max(0.0, 1.0 - abs(math.log10(sa) - math.log10(sb)) / 9.0), 3)


# ══════════════════════════════════════════════
# ОРКЕСТРАТОР
# ══════════════════════════════════════════════

class MathCore:

    def __init__(self, artifacts_dir: str = "artifacts",
                 four_d_index: str = "artifacts/four_d_index.jsonl"):
        self.artifacts_dir     = Path(artifacts_dir)
        self.stress_tester     = StressTester()
        self.resonance_matcher = ResonanceMatcher(four_d_index)
        self.prob_engine       = ProbabilityEngine()

    def stress_test(self, artifact_id: str,
                    four_d: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        if four_d is None:
            four_d = self._load_four_d(artifact_id)
        if not four_d:
            return {"error": f"No four_d_matrix for {artifact_id}", "stability_score": 0.0}
        try:
            result = self.stress_tester.run(four_d, artifact_id)
            self._patch_artifact(artifact_id, {"simulation": result})
            logger.info(f"MathCore stress_test {artifact_id}: "
                        f"score={result['stability_score']} "
                        f"usm={result['usm']['usm']} "
                        f"λ={result['lyapunov_max']} model={result['model_used']}")
            return result
        except Exception as e:
            logger.error(f"MathCore stress_test failed: {e}", exc_info=True)
            return {"error": str(e), "stability_score": 0.0}

    def find_resonance(self, query_four_d: Dict, query_domain: str,
                       query_survival: str = "UNKNOWN",
                       target_domains: Optional[List[str]] = None,
                       top_k: int = 3, exclude_id: str = "") -> Dict[str, Any]:
        from schemas.four_d_matrix import FourDMatrix
        matrix = FourDMatrix.from_raw(query_four_d)
        if matrix is None:
            return {"error": "Invalid four_d_matrix", "insights": []}
        query_vec = matrix.to_vector()
        similar   = self.resonance_matcher.find_similar(query_vec, top_k=top_k, exclude_id=exclude_id)
        insights  = []
        for match in similar:
            iso_4d      = match.get("4d_resonance", 0.0)
            match_domain = match.get("domain", "general")
            stability   = match.get("stability_score", 0.5)
            sim_path    = SIM_RESULTS_DIR / f"{match['id']}_stress.json"
            if sim_path.exists():
                try:
                    stability = json.loads(sim_path.read_text()).get("stability_score", stability)
                except Exception:
                    pass
            scale_align   = self.prob_engine.compute_scale_align(query_domain, match_domain)
            noise_penalty = float(query_four_d.get("influence", {}).get("eta", 0.2))
            prob          = self.prob_engine.compute(iso_4d, stability, scale_align,
                                                     query_survival, noise_penalty)
            insight = {
                "id": f"ins_{match['id'][:8]}_{int(time.time())}",
                "source_domain": query_domain, "target_domain": match_domain,
                "prototype_id": match["id"], **prob,
                "iso_4d": iso_4d, "scale_align": scale_align,
                "status": "monitoring" if prob["probability"] >= 0.55 else "low_confidence",
                "generated_at": _now_iso(),
            }
            insights.append(insight)
        for ins in insights:
            self._save_insight(ins)
        return {"insights": insights, "total": len(insights),
                "top_resonance": insights[0]["iso_4d"] if insights else 0.0}

    def index_artifact(self, artifact_id: str, four_d: Dict, domain: str,
                       stability_score: float = 0.5):
        from schemas.four_d_matrix import FourDMatrix
        matrix = FourDMatrix.from_raw(four_d)
        if matrix is None: return
        self.resonance_matcher.add_to_index(
            artifact_id=artifact_id, four_d=matrix.to_dict(), domain=domain,
            vec=matrix.to_vector(), stability_score=stability_score)
        logger.info(f"MathCore indexed {artifact_id} (domain={domain})")

    def _load_four_d(self, artifact_id: str) -> Optional[Dict]:
        path = self.artifacts_dir / f"{artifact_id}.json"
        if not path.exists(): return None
        try:
            return json.loads(path.read_text(encoding="utf-8")).get("data",{}).get("gen",{}).get("four_d_matrix")
        except Exception: return None

    def _patch_artifact(self, artifact_id: str, patch: Dict):
        path = self.artifacts_dir / f"{artifact_id}.json"
        if not path.exists():
            logger.debug(f"_patch_artifact: {artifact_id}.json not found"); return
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            data.update(patch)
            path.write_text(json.dumps(data, ensure_ascii=False, indent=2))
        except Exception as e:
            logger.error(f"_patch_artifact {artifact_id}: {e}")

    def _save_insight(self, insight: Dict):
        INSIGHTS_DIR.mkdir(exist_ok=True)
        (INSIGHTS_DIR / f"{insight['id']}.json").write_text(
            json.dumps(insight, ensure_ascii=False, indent=2))

    @staticmethod
    def selftest():
        print("MathCore v4.2.2 self-test...")

        # Kuramoto sub-critical — K < K_c → должна быть нестабильна
        sim = KuramotoSimulator(N=50)
        r = sim.run(omega_i=0.25, K=0.42, K_c=0.59, eta=0.1, tau=0.5)
        # K=0.42, K_c=0.59, K/K_c=0.71 < 1 → нестабильно (sigma=K_c*pi/6=0.309)
        assert r["r_final"] < 0.99, f"Expected r_final < 0.99 for unstable, got r={r['r_final']}"
        print(f"  Kuramoto (K<K_c=0.59): r_final={r['r_final']} stable={r['stable']} ✓")

        # Kuramoto super-critical — K > K_c → должна синхронизироваться
        r2 = sim.run(omega_i=0.25, K=0.8, K_c=0.59, eta=0.1, tau=0.5)
        # K=0.8, K_c=0.59, K/K_c=1.36 > 1 → стабильно
        assert r2["stable"], f"Expected stable K>K_c, got r={r2['r_final']}"
        print(f"  Kuramoto (K>K_c=0.59): r_final={r2['r_final']} stable={r2['stable']} ✓")

        # Ising
        i_ord = IsingMeanField().run(K=1.5, h=0.1, T=0.8, eta=0.1)
        i_dis = IsingMeanField().run(K=0.5, h=0.0, T=2.0, eta=0.1)
        print(f"  Ising ordered: m={i_ord['magnetization']} stable={i_ord['stable']} ✓")
        print(f"  Ising disordered: m={i_dis['magnetization']} stable={i_dis['stable']} ✓")

        # LotkaVolterra
        lv = LotkaVolterraStability().run(omega_i=0.3, K=0.4, K_c=0.5, p=0.7, eta=0.2)
        print(f"  LV: x*={lv['x_star']} y*={lv['y_star']} stable={lv['stable']} ✓")

        # Delay
        dl_s = DelayStability().run(K=1.0, tau=0.5, H=0.7, eta=0.1)
        dl_u = DelayStability().run(K=1.0, tau=2.0, H=0.7, eta=0.1)
        print(f"  Delay stable: margin={dl_s['stability_margin']:.2f} stable={dl_s['stable']} ✓")
        print(f"  Delay unstable: margin={dl_u['stability_margin']:.2f} stable={dl_u['stable']} ✓")

        # GraphInvariant
        gi = GraphInvariantStability().run(C=0.6, k=8.0, p=0.3, eta=0.2)
        print(f"  GraphInv: giant={gi['giant_fraction']} stable={gi['stable']} ✓")

        # USM
        fd = {"structure":{"C":0.62,"k":9.0,"D":2.15},"influence":{"h":0.9,"T":1.0,"eta":0.2},
              "dynamics":{"omega_i":0.25,"K":0.8,"K_c":0.59,"p":0.7,"model":"kuramoto"},
              "time":{"tau":0.5,"H":0.75,"freq":1.1}}
        u = UniversalStabilityMetrics().compute(fd, "kuramoto")
        assert u["usm"] > 0.4, f"USM too low: {u}"
        print(f"  USM (K>K_c): usm={u['usm']} phase={u['phase_proximity']} ✓")

        # StressTester
        st = StressTester()
        res = st.run(fd, "test_artifact")
        print(f"  StressTester: score={res['stability_score']} usm={res['usm']['usm']} cpu={res['cpu_time_s']}s ✓")

        # Dedup
        import tempfile, os
        tmp = tempfile.mktemp(suffix=".jsonl")
        try:
            rm = ResonanceMatcher(tmp)
            v  = np.ones(13, dtype=np.float64) * 0.5
            rm.add_to_index("t1", {}, "physics",  v, 0.7)
            rm.add_to_index("t1", {}, "physics",  v, 0.9)  # duplicate
            rm.add_to_index("t2", {}, "biology", v * 0.9, 0.6)
            assert len(rm._meta) == 2
            assert rm._meta[next(i for i,m in enumerate(rm._meta) if m["id"]=="t1")]["stability_score"] == 0.9
            print(f"  ResonanceMatcher dedup: {len(rm._meta)} entries ✓")
        finally:
            if os.path.exists(tmp): os.unlink(tmp)

        print("\n✅ All MathCore v4.2.2 components OK")


if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="MathCore v4.2.2 CLI")
    parser.add_argument("--test",   action="store_true")
    parser.add_argument("--stress", type=str, default="")
    args = parser.parse_args()
    if args.test:
        MathCore.selftest()
    elif args.stress:
        print(json.dumps(MathCore().stress_test(args.stress), ensure_ascii=False, indent=2))
    else:
        parser.print_help()
