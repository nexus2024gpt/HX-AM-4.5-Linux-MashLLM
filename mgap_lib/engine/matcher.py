# mgap_lib/engine/matcher.py — MGAP Library v1.0
"""
MGAPEngine — основной оркестратор MGAP Library.

Полный цикл для одного артефакта:
  1. Загрузка артефакта (из файла или JSON)
  2. Классификация домена (DomainClassifier — без LLM)
  3. Фильтрация моделей (по math_type и/или сектору)
  4. Вычисление 4D-резонанса (compute_4d_resonance из schemas)
  5. Вычисление Gap (GapCalculator)
  6. Перевод параметров (translation_map)
  7. Генерация кода адаптации
  8. Расчёт на примере (example_data)
  9. Улучшение blind_spot (опционально LLM)
  10. Сохранение в БД и/или JSON

Отличие от mgap_matcher.py (v4.4):
  - Использует DomainClassifier для UNESCO-иерархии
  - Сохраняет результаты в БД (ArtifactRun)
  - Принимает фильтры по sector_code
  - Полностью совместим с mgap_matcher.py по выходному формату

Пример:
    engine = MGAPEngine.from_json("mgap_registry.json")
    results = engine.match_artifact("32d4aa917ac4", top_k=3)
    for r in results:
        print(r["model_id"], r["resonance"], r["gap"]["risk_level"])
"""

from __future__ import annotations

import json
import logging
import math
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from mgap_lib.engine.domain_classifier import DomainClassifier, DomainClassificationResult
from mgap_lib.engine.gap_calculator import GapCalculator, GapComponents
from mgap_lib.engine.registry import RegistryLoader

logger = logging.getLogger("MGAP.engine")

# ── Нормализация math_type ────────────────────────────────────────────────────
_MATH_TYPE_ALIASES: Dict[str, str] = {
    "delay_ode":       "delay",
    "delay-ode":       "delay",
    "graph-invariant": "graph_invariant",
}


def _norm_mt(t: str) -> str:
    return _MATH_TYPE_ALIASES.get(t.lower().strip(), t.lower().strip())


# ══════════════════════════════════════════════════════════════════════════════
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ (совместимость с mgap_matcher.py v4.4)
# ══════════════════════════════════════════════════════════════════════════════

def _flat_4d(four_d: Dict) -> Dict[str, Any]:
    dyn = four_d.get("dynamics", {})
    inf = four_d.get("influence", {})
    tim = four_d.get("time", {})
    return {
        "tau":     float(tim.get("tau",     0.5)),
        "K":       float(dyn.get("K",       0.35)),
        "K_c":     float(dyn.get("K_c",     0.48)),
        "eta":     float(inf.get("eta",     0.2)),
        "omega_i": float(dyn.get("omega_i", 0.25)),
        "p":       float(dyn.get("p",       0.65)),
        "model":   str(dyn.get("model",     "kuramoto")),
    }


def _art_vector(four_d: Dict) -> Optional[np.ndarray]:
    try:
        from schemas.four_d_matrix import FourDMatrix
        m = FourDMatrix.from_raw(four_d)
        return m.to_vector() if m else None
    except Exception:
        return None


def _model_vector(model: Dict) -> Optional[np.ndarray]:
    return _art_vector(model.get("four_d_matrix") or {})


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
    type_bonus = 0.3 if _norm_mt(model.get("math_type", "")) == _norm_mt(art_math) else 0.0
    return round(vec_res * 0.7 + type_bonus, 3)


def _resonance_fallback(flat: Dict, model: Dict) -> float:
    m4d    = model.get("four_d_matrix") or {}
    m_flat = _flat_4d(m4d)
    ranges = model.get("expected_ranges") or {}
    weights = model.get("weights") or {}
    total = score = 0.0
    for key in ("tau", "K", "eta"):
        r   = ranges.get(key, [0.0, 1.0])
        if isinstance(r, list) and len(r) == 2:
            lo, hi = r
        else:
            lo, hi = 0.0, 1.0
        w    = float(weights.get(key, 1.0))
        span = max(hi - lo, 1e-9)
        sim  = max(0.0, 1.0 - abs(flat.get(key, 0.5) - m_flat.get(key, 0.5)) / span)
        score += sim * w
        total += w
    return round(score / max(total, 1e-9), 3)


def _extract_thresholds(artifact: Dict, model: Dict) -> Dict:
    sim   = artifact.get("simulation") or {}
    ver   = artifact.get("data", {}).get("ver", {}) or {}
    stress = ver.get("stress_test") or {}
    ct    = model.get("critical_thresholds") or {}

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


# ══════════════════════════════════════════════════════════════════════════════
# КОДОГЕНЕРАЦИЯ (из mgap_matcher.py v4.4 — без изменений)
# ══════════════════════════════════════════════════════════════════════════════

def _generate_code(model: Dict, thresholds: Dict, flat: Dict) -> str:
    mt   = _norm_mt(model.get("math_type", "kuramoto"))
    eta_c = thresholds["eta_critical"]
    tau_c = thresholds["tau_robustness"]
    prog  = (model.get("programs") or ["target_system"])[0]

    if mt == "graph_invariant":
        K_min = model.get("critical_thresholds", {}).get("K_min", 0.3)
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
    return f"# math_type '{mt}' — code generation not yet implemented"


def _calculate_example(model: Dict, thresholds: Dict) -> Dict:
    example = model.get("example_data") or {}
    eta_c = thresholds["eta_critical"]
    tau_c = thresholds["tau_robustness"]
    t     = example.get("type", "graph_invariant")

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
        if K < K_c:      warns.append(f"K={K:.3f} < K_c={K_c:.3f}")
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
    return {"error": f"unknown example type: {t}"}


# ══════════════════════════════════════════════════════════════════════════════
# MGAPEngine
# ══════════════════════════════════════════════════════════════════════════════

class MGAPEngine:
    """
    Оркестратор MGAP Library v1.0.

    Использует RegistryLoader, DomainClassifier, GapCalculator.
    Совместим с mgap_matcher.py v4.4 по выходному формату (ключи идентичны).
    Добавляет: gap, domain_classification, disc_code, sector_code.
    """

    def __init__(
        self,
        registry: Optional[RegistryLoader] = None,
        classifier: Optional[DomainClassifier] = None,
        gap_calc: Optional[GapCalculator] = None,
        artifacts_dir: Optional[Path] = None,
        db_session=None,
        llm=None,
        results_dir: Optional[Path] = None,
        gap_mode: str = "max",
    ):
        self.registry     = registry or RegistryLoader(db_session=db_session)
        self.classifier   = classifier or DomainClassifier()
        self.gap_calc     = gap_calc or GapCalculator()
        self.artifacts_dir = artifacts_dir or Path("artifacts")
        self.llm          = llm or self._try_load_llm()
        self.results_dir  = results_dir or Path("mgap_results")
        self.gap_mode     = gap_mode

    @classmethod
    def from_json(
        cls,
        registry_path: str = "mgap_registry.json",
        artifacts_dir: str = "artifacts",
        use_llm: bool = True,
        gap_mode: str = "max",
    ) -> "MGAPEngine":
        """Быстрый конструктор из JSON-файла (без БД)."""
        registry = RegistryLoader(registry_path=Path(registry_path))
        llm = cls._try_load_llm_static() if use_llm else None
        return cls(
            registry=registry,
            artifacts_dir=Path(artifacts_dir),
            llm=llm,
            gap_mode=gap_mode,
        )

    @staticmethod
    def _try_load_llm():
        try:
            from llm_client_v_4 import LLMClient
            return LLMClient()
        except Exception:
            return None

    @staticmethod
    def _try_load_llm_static():
        return MGAPEngine._try_load_llm()

    # ── публичный API ─────────────────────────────────────────────────────────

    def match_artifact(
        self,
        artifact_id: str,
        top_k: int = 3,
        math_type_only: bool = True,
        model_id: Optional[str] = None,
        sector_filter: Optional[str] = None,
        save_to_db: bool = False,
        save_to_json: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Основной метод: сопоставляет артефакт с моделями реестра.

        Returns:
            Список Dict, совместимый с форматом mgap_matcher.py v4.4.
            Дополнительные поля: gap, domain_classification.
        """
        artifact = self._load_artifact(artifact_id)
        if not artifact:
            return [{"error": f"Artifact '{artifact_id}' not found",
                     "artifact_id": artifact_id}]

        four_d = self._extract_four_d(artifact)
        if not four_d:
            return [{"error": "No four_d_matrix — run migrate_to_v42.py first",
                     "artifact_id": artifact_id}]

        flat     = _flat_4d(four_d)
        art_math = _norm_mt(flat["model"])
        art_vec  = _art_vector(four_d)

        # Классификация домена
        raw_domain  = artifact.get("data", {}).get("domain", "general") or "general"
        domain_cls  = self.classifier.classify(raw_domain)

        # Фильтрация кандидатов
        candidates = self.registry.get_all()
        if model_id:
            candidates = [m for m in candidates if m.get("id") == model_id]
        if math_type_only:
            candidates = [m for m in candidates
                          if _norm_mt(m.get("math_type", "")) == art_math]
        if sector_filter:
            candidates = [m for m in candidates if m.get("sector_code") == sector_filter]

        if not candidates:
            return [{"error": f"No candidates (math_type={art_math}, math_type_only={math_type_only})",
                     "artifact_id": artifact_id}]

        # Сортировка по резонансу
        scored: List[Tuple[float, Dict]] = []
        for model in candidates:
            res = _compute_resonance(art_vec, model, art_math)
            scored.append((res, model))
        scored.sort(key=lambda x: -x[0])

        results = []
        for resonance, model in scored[:top_k]:
            thresholds = _extract_thresholds(artifact, model)
            match_dict = self._build_match(
                artifact_id=artifact_id,
                artifact=artifact,
                four_d=four_d,
                flat=flat,
                thresholds=thresholds,
                art_math=art_math,
                model=model,
                resonance=resonance,
                domain_cls=domain_cls,
            )
            results.append(match_dict)

            if save_to_db:
                self._save_run_to_db(match_dict)

        if save_to_json and results:
            self._save_to_json(artifact_id, results)

        return results

    def match_batch(
        self,
        top_k: int = 2,
        math_type_only: bool = True,
        min_resonance: float = 0.3,
        save_to_db: bool = False,
    ) -> Dict[str, List[Dict]]:
        """Прогоняет все артефакты из artifacts/ через MGAPEngine."""
        out: Dict[str, List[Dict]] = {}
        if not self.artifacts_dir.exists():
            return out
        for f in sorted(self.artifacts_dir.glob("*.json")):
            if f.stem == "invariant_graph" or ".hyx-portal" in f.name:
                continue
            art_id = f.stem
            try:
                matches = self.match_artifact(art_id, top_k=top_k,
                                               math_type_only=math_type_only,
                                               save_to_db=save_to_db,
                                               save_to_json=False)
                ok = [m for m in matches
                      if not m.get("error") and m.get("resonance", 0) >= min_resonance]
                if ok:
                    out[art_id] = ok
            except Exception as e:
                logger.warning(f"match_batch: {art_id} failed — {e}")
        return out

    # ── сборка результата ─────────────────────────────────────────────────────

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
        domain_cls: DomainClassificationResult,
    ) -> Dict[str, Any]:
        math_match = _norm_mt(model.get("math_type", "")) == art_math

        # Gap
        gap_obj = self.gap_calc.compute(
            {"eta": flat["eta"], "tau": flat["tau"], "K": flat["K"]},
            model.get("critical_thresholds") or {},
            mode=self.gap_mode,
        )

        # Перевод параметров
        translation = self._translate_params(flat, thresholds, model)

        # Слепая зона
        raw_blind = (model.get("blind_spot_template") or "").format(
            eta_max=thresholds["eta_critical"],
            tau_max=thresholds["tau_robustness"],
        )
        blind_spot = self._improve_blind_spot(raw_blind, model)

        # Код
        code_snippet = _generate_code(model, thresholds, flat)
        # Пример
        calculation  = _calculate_example(model, thresholds)

        # Вердикт
        verdict = self._build_verdict(model, gap_obj, resonance, thresholds)

        # Краткая сводка артефакта
        gen      = artifact.get("data", {}).get("gen", {})
        archivist = artifact.get("archivist") or {}

        return {
            "artifact_id":     artifact_id,
            "model_id":        model.get("id"),
            "model_name":      model.get("name"),
            "logia":           model.get("logia"),
            "industry":        model.get("industry"),
            "programs":        model.get("programs", []),
            "disc_code":       model.get("disc_code") or domain_cls.disc_code,
            "sector_code":     model.get("sector_code") or domain_cls.sector_code,
            "resonance":       resonance,
            "math_type_match": math_match,
            "gap":             gap_obj.to_dict(),
            "domain_classification": domain_cls.to_dict(),
            "artifact_summary": {
                "domain":           artifact.get("data", {}).get("domain", "—"),
                "hypothesis":       gen.get("hypothesis", "")[:120],
                "math_type":        art_math,
                "stability_score":  thresholds["stability_score"],
                "survival_verified": thresholds["survival_verified"],
                "novelty":          archivist.get("novelty", "—"),
                "disc_name_ru":     domain_cls.disc_name_ru,
                "sector_name_ru":   domain_cls.sector_name_ru,
            },
            "thresholds":    thresholds,
            "translation":   translation,
            "blind_spot":    blind_spot,
            "adaptation": {
                "formula":      model.get("math_adaptation_formula", "—"),
                "code_snippet": code_snippet,
                "programs":     model.get("programs", []),
            },
            "calculation":   calculation,
            "verdict":       verdict,
            "generated_at":  datetime.now(timezone.utc).isoformat(),
        }

    # ── helpers ───────────────────────────────────────────────────────────────

    def _translate_params(self, flat: Dict, thresholds: Dict, model: Dict) -> Dict:
        tmap = model.get("translation_map") or {}
        result: Dict = {}
        for key, val in [("tau", flat["tau"]), ("K", flat["K"]), ("eta", flat["eta"])]:
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

    def _build_verdict(self, model: Dict, gap: GapComponents, resonance: float, thresholds: Dict) -> Dict:
        warn = gap.is_warning
        prog = (model.get("programs") or ["target_system"])[0]
        if warn:
            biz_rec = (f"Система НА ПОРОГЕ нестабильности (risk={gap.risk_level}). "
                       f"Внедрить MGAP-монитор в {prog}. "
                       f"Снижение риска каскадных отказов: 15–25%.")
            dev_act = f"Добавить mgap_stability_monitor() в {prog}"
        else:
            biz_rec = f"Система стабильна (risk={gap.risk_level}). Мониторинг полезен профилактически."
            dev_act = f"Добавить пассивный мониторинг порогов в {prog}"
        return {
            "verdict": "Применимо как расширение" if warn else "Применимо, мониторинг",
            "for_developer": {
                "action":             dev_act,
                "code_reference":     "adaptation.code_snippet",
                "new_config_params": {
                    "eta_critical":   thresholds["eta_critical"],
                    "tau_robustness": thresholds["tau_robustness"],
                },
                "gap_summary":        gap.to_dict(),
            },
            "for_business": {
                "summary":         (f"Артефакт резонирует с моделью «{model.get('name')}» "
                                    f"({model.get('logia')}, resonance={resonance:.2f})."),
                "blind_spot":      (model.get("blind_spot_template") or "—").format(
                                       eta_max=thresholds["eta_critical"],
                                       tau_max=thresholds["tau_robustness"]),
                "recommendation":  biz_rec,
                "stability_score": thresholds.get("stability_score", "—"),
                "estimated_roi":   "Снижение риска каскадных отказов на 15–25%",
            },
        }

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

    @staticmethod
    def _extract_four_d(artifact: Dict) -> Optional[Dict]:
        return artifact.get("data", {}).get("gen", {}).get("four_d_matrix")

    def _save_to_json(self, artifact_id: str, results: List[Dict]):
        self.results_dir.mkdir(exist_ok=True)
        ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = self.results_dir / f"{artifact_id}_{ts}.json"
        try:
            path.write_text(json.dumps({"artifact_id": artifact_id, "matches": results},
                                       ensure_ascii=False, indent=2))
        except Exception as e:
            logger.warning(f"_save_to_json failed: {e}")

    def _save_run_to_db(self, match: Dict):
        try:
            from mgap_lib.models.database import ArtifactRun, get_session
            session = get_session()
            run = ArtifactRun(
                artifact_id = match.get("artifact_id", ""),
                model_id    = match.get("model_id", ""),
                domain      = match.get("artifact_summary", {}).get("domain"),
                disc_code   = match.get("disc_code"),
                sector_code = match.get("sector_code"),
                resonance   = match.get("resonance"),
                risk_level  = match.get("gap", {}).get("risk_level"),
                status      = "ok",
            )
            run.results = match
            session.add(run)
            session.commit()
            session.close()
        except Exception as e:
            logger.warning(f"_save_run_to_db failed: {e}")

    # ── реестр ────────────────────────────────────────────────────────────────

    def get_registry_summary(self) -> List[Dict]:
        return self.registry.get_summary()
