# four_d_builder.py — HX-AM v4.6
"""
Программное построение four_d_matrix и b_sync
без участия LLM.

Принцип: 4D-матрица — это вычисление по правилам,
а не творческая задача. LLM выбирает только model (1 слово),
всё остальное строится детерминированно из домена,
модели динамики и текста гипотезы.

Разнообразие обеспечивается:
  - доменными диапазонами (разные начальные точки)
  - текстовым сигналом (числа, ключевые слова)
  - семантической калибровкой по архиву
  - детерминированным джиттером (hash текста → seed)
"""

import hashlib
import logging
import math
import re
from typing import Dict, List, Optional, Tuple

import numpy as np

from domain_config import NEW_DOMAIN_PARAMS

logger = logging.getLogger("HXAM.4dbuilder")

# ── Типичные диапазоны параметров по домену ───────────────────
# (min, max) для каждого параметра
DOMAIN_PARAMS: Dict[str, Dict] = {
    "biology":      {"k":(6,12),  "C":(0.60,0.85), "D":(1.8,2.3), "tau":(1.0,5.0),  "H":(0.55,0.75), "freq":(0.5,2.0),  "h":(0.5,2.0), "T":(0.8,1.5),  "eta":(0.15,0.35)},
    "neuroscience": {"k":(8,15),  "C":(0.65,0.90), "D":(2.0,2.8), "tau":(0.1,1.0),  "H":(0.60,0.80), "freq":(2.0,8.0),  "h":(1.0,2.5), "T":(0.7,1.3),  "eta":(0.10,0.30)},
    "physics":      {"k":(4,8),   "C":(0.40,0.70), "D":(1.5,3.5), "tau":(0.1,2.0),  "H":(0.45,0.65), "freq":(1.0,8.0),  "h":(0.5,3.0), "T":(0.5,2.5),  "eta":(0.10,0.25)},
    "chemistry":    {"k":(3,8),   "C":(0.35,0.65), "D":(1.5,2.5), "tau":(0.1,1.5),  "H":(0.45,0.65), "freq":(1.0,6.0),  "h":(0.5,2.5), "T":(0.8,3.0),  "eta":(0.10,0.30)},
    "economics":    {"k":(15,40), "C":(0.35,0.65), "D":(2.5,3.2), "tau":(2.0,8.0),  "H":(0.55,0.80), "freq":(0.2,1.5),  "h":(1.5,3.5), "T":(1.2,2.5),  "eta":(0.25,0.55)},
    "sociology":    {"k":(12,30), "C":(0.50,0.75), "D":(2.2,2.9), "tau":(1.0,5.0),  "H":(0.50,0.72), "freq":(0.3,1.5),  "h":(1.0,3.0), "T":(1.0,2.0),  "eta":(0.20,0.45)},
    "ecology":      {"k":(5,10),  "C":(0.55,0.80), "D":(1.8,2.5), "tau":(2.0,10.0), "H":(0.60,0.80), "freq":(0.1,1.0),  "h":(1.0,2.5), "T":(1.0,2.0),  "eta":(0.20,0.40)},
    "psychology":   {"k":(8,20),  "C":(0.50,0.80), "D":(2.0,2.7), "tau":(0.5,3.0),  "H":(0.55,0.75), "freq":(0.5,3.0),  "h":(0.8,2.5), "T":(1.0,2.0),  "eta":(0.15,0.40)},
    "geology":      {"k":(4,9),   "C":(0.40,0.70), "D":(2.0,3.0), "tau":(5.0,20.0), "H":(0.65,0.85), "freq":(0.05,0.5), "h":(1.5,3.5), "T":(1.0,2.5),  "eta":(0.25,0.50)},
    "linguistics":  {"k":(10,25), "C":(0.55,0.80), "D":(2.0,2.8), "tau":(0.5,3.0),  "H":(0.50,0.70), "freq":(0.5,3.0),  "h":(0.5,2.0), "T":(0.8,1.8),  "eta":(0.15,0.35)},
    "medicine":     {"k":(6,14),  "C":(0.60,0.85), "D":(1.8,2.5), "tau":(0.5,5.0),  "H":(0.55,0.78), "freq":(0.5,4.0),  "h":(1.0,2.8), "T":(0.8,1.8),  "eta":(0.15,0.35)},
    "astronomy":    {"k":(4,10),  "C":(0.30,0.60), "D":(2.0,3.5), "tau":(5.0,20.0), "H":(0.60,0.85), "freq":(0.05,1.0), "h":(0.5,3.0), "T":(0.5,2.0),  "eta":(0.10,0.30)},
    "general":      {"k":(6,15),  "C":(0.50,0.75), "D":(2.0,2.5), "tau":(1.0,5.0),  "H":(0.55,0.72), "freq":(0.5,2.0),  "h":(0.8,2.0), "T":(0.9,1.8),  "eta":(0.18,0.38)},
}
DOMAIN_PARAMS.update(NEW_DOMAIN_PARAMS)

# ── Типичные параметры динамики по модели ─────────────────────
# K_ratio: K = K_c * K_ratio  (обеспечивает K > K_c)
MODEL_DYNAMICS: Dict[str, Dict] = {
    "kuramoto":       {"K_c":(0.40,0.70), "K_ratio":(1.15,1.45), "omega_i":(0.20,1.50), "p":(0.60,0.85)},
    "percolation":    {"K_c":(0.30,0.55), "K_ratio":(1.10,1.40), "omega_i":(0.40,1.20), "p":(0.50,0.80)},
    "ising":          {"K_c":(0.55,0.80), "K_ratio":(1.10,1.35), "omega_i":(0.10,0.60), "p":(0.55,0.80)},
    "lotka_volterra": {"K_c":(0.25,0.50), "K_ratio":(1.10,1.40), "omega_i":(0.08,0.40), "p":(0.55,0.75)},
    "delay":          {"K_c":(0.30,0.55), "K_ratio":(1.10,1.35), "omega_i":(0.15,0.60), "p":(0.60,0.80)},
    "graph_invariant":{"K_c":(0.35,0.60), "K_ratio":(1.10,1.40), "omega_i":(0.12,0.50), "p":(0.60,0.85)},
}

# ── Ключевые слова → выбор модели динамики ────────────────────
MODEL_KEYWORDS: Dict[str, List[str]] = {
    "kuramoto":        ["синхрон","осциллятор","фаза","ритм","coupling","phase","sync","частота","пейсмейкер","мерцание","coherence"],
    "percolation":     ["каскад","перколяц","порог","эпидем","распростран","заражен","threshold","cascade","epidemic","связность"],
    "ising":           ["бинарн","спин","мнение","консенсус","голосован","phase transition","намагнич","binary","opinion"],
    "lotka_volterra":  ["хищник","жертва","конкуренц","популяц","predator","prey","competition","вымирани","ресурс"],
    "delay":           ["запаздыван","задержк","лаг","обратная связь","feedback","delay","гормон","цикл","мультипликатор"],
    "graph_invariant": ["граф","топологи","связность","хаб","кластер","инвариант","community","hub","topology"],
}


class FourDBuilder:
    """
    Строит four_d_matrix программно из текста и домена.
    Не требует LLM для числовых полей.
    """

    def detect_model(self, hypothesis: str, mechanism: str) -> str:
        """Выбор модели по ключевым словам в тексте."""
        text = (hypothesis + " " + mechanism).lower()
        scores: Dict[str, int] = {m: 0 for m in MODEL_KEYWORDS}
        for model, keywords in MODEL_KEYWORDS.items():
            for kw in keywords:
                if kw in text:
                    scores[model] += 1
        best = max(scores, key=lambda m: scores[m])
        if scores[best] == 0:
            return "kuramoto"  # default
        logger.debug(f"FourDBuilder: detected model={best} scores={scores}")
        return best

    def _seed(self, text: str) -> int:
        """Детерминированный seed из текста → разнообразие без случайности."""
        return int(hashlib.md5(text[:100].encode()).hexdigest()[:8], 16)

    def _extract_text_numbers(self, text: str) -> List[float]:
        """Извлечение числовых подсказок из текста гипотезы."""
        # Ищем проценты, доли, коэффициенты упомянутые в тексте
        matches = re.findall(
            r'\b(\d+(?:\.\d+)?)\s*(?:%|процент|доля|ratio|коэффициент)',
            text.lower()
        )
        result = []
        for m in matches:
            v = float(m)
            if v > 1:
                v = v / 100.0  # конвертируем % в долю
            result.append(min(1.0, v))
        return result[:3]  # не более 3 числовых подсказок

    def _lerp(self, lo: float, hi: float, t: float) -> float:
        """Линейная интерполяция."""
        return round(lo + (hi - lo) * t, 3)

    def build(
        self,
        domain: str,
        hypothesis: str,
        mechanism: str,
        model: Optional[str] = None,
    ) -> Dict:
        """
        Строит полную four_d_matrix.

        Args:
            domain:     нормализованный домен
            hypothesis: текст гипотезы
            mechanism:  текст механизма
            model:      если None — определяется автоматически

        Returns:
            dict совместимый с FourDMatrix.from_raw()
        """
        # 1. Выбор модели динамики
        if not model:
            model = self.detect_model(hypothesis, mechanism)

        # 2. Детерминированный jitter из текста (0.0 - 1.0)
        rng   = np.random.default_rng(self._seed(hypothesis + domain))
        nums  = self._extract_text_numbers(hypothesis + " " + mechanism)
        jitter = float(rng.uniform(0.15, 0.85))  # базовая позиция в диапазоне

        # Если в тексте есть числа — используем их как сигнал
        if nums:
            jitter = float(np.clip(np.mean(nums), 0.15, 0.85))

        # 3. Доменные параметры
        dp = DOMAIN_PARAMS.get(domain, DOMAIN_PARAMS["general"])

        def pick(key: str, fallback_lo: float, fallback_hi: float) -> float:
            lo, hi = dp.get(key, (fallback_lo, fallback_hi))
            # Добавляем небольшое дополнительное разнообразие
            local_j = float(rng.uniform(0.1, 0.9))
            return self._lerp(lo, hi, (jitter * 0.7 + local_j * 0.3))

        # 4. Параметры динамики (с гарантией K > K_c)
        md   = MODEL_DYNAMICS.get(model, MODEL_DYNAMICS["kuramoto"])
        K_c  = self._lerp(*md["K_c"],   float(rng.uniform(0.2, 0.8)))
        K    = round(K_c * self._lerp(*md["K_ratio"], float(rng.uniform(0.3, 0.7))), 3)
        K    = min(K, 2.0)  # hard cap из схемы

        four_d = {
            "structure": {
                "C": pick("C",   0.40, 0.80),
                "k": round(self._lerp(*dp.get("k", (6, 15)), jitter), 1),
                "D": pick("D",   1.5,  2.8),
            },
            "influence": {
                "h":   pick("h",   0.5, 2.5),
                "T":   pick("T",   0.8, 2.0),
                "eta": pick("eta", 0.1, 0.5),
            },
            "dynamics": {
                "omega_i": self._lerp(*md["omega_i"], float(rng.uniform(0.2, 0.8))),
                "K":       K,
                "K_c":     K_c,
                "p":       self._lerp(*md["p"], float(rng.uniform(0.25, 0.75))),
                "model":   model,
            },
            "time": {
                "tau":  pick("tau",  0.5, 10.0),
                "H":    pick("H",    0.45, 0.80),
                "freq": pick("freq", 0.3, 5.0),
            },
        }

        # 5. Валидация через существующую схему
        try:
            from schemas.four_d_matrix import FourDMatrix
            matrix = FourDMatrix.from_raw(four_d)
            if matrix is not None:
                result = matrix.to_dict()
                logger.info(
                    f"FourDBuilder: built 4D model={model} domain={domain} "
                    f"K={result['dynamics']['K']:.3f} K_c={result['dynamics']['K_c']:.3f}"
                )
                return result
        except Exception as e:
            logger.warning(f"FourDBuilder: schema validation failed — {e}")

        return four_d


# ── b_sync вычисляется семантически ───────────────────────────

def compute_b_sync(
    domain: str,
    hypothesis: str,
    mechanism: str,
    space,   # SemanticSpace
) -> float:
    """
    Вычисляет b_sync программно через семантическое пространство.

    b_sync = насколько хорошо механизм переносится между доменами.

    Компоненты:
      domain_diversity  (0.4) — насколько разные домены у ближайших соседей
      specificity       (0.4) — насколько гипотеза специфична для своего домена
      model_universality(0.2) — универсальность выбранной модели динамики
    """
    UNIVERSAL_MODELS = {"kuramoto", "graph_invariant", "percolation"}

    # Проверяем что space не пустой
    if not space.vectors:
        return 0.60  # default для первых артефактов

    vec      = space.encode(hypothesis)
    similar  = space.nearest(hypothesis, top_k=5, threshold=0.40)
    spec     = space.specificity(vec, domain)

    # Разнообразие доменов среди соседей
    if similar:
        neighbor_domains = [s.get("domain", domain) for s in similar]
        unique_domains   = len(set(neighbor_domains))
        own_domain_count = sum(1 for d in neighbor_domains if d == domain)
        # Чем больше соседей из других доменов — тем выше переносимость
        diversity = (len(similar) - own_domain_count) / max(len(similar), 1)
    else:
        diversity = 0.5  # нет соседей = неизвестно

    # Универсальность модели динамики
    builder   = FourDBuilder()
    model     = builder.detect_model(hypothesis, mechanism)
    model_u   = 0.75 if model in UNIVERSAL_MODELS else 0.45

    b_sync = round(
        0.40 * diversity
        + 0.40 * float(spec)
        + 0.20 * model_u,
        3
    )
    b_sync = max(0.10, min(0.98, b_sync))

    logger.debug(
        f"b_sync={b_sync:.3f} "
        f"diversity={diversity:.2f} spec={spec:.3f} model_u={model_u:.2f}"
    )
    return b_sync