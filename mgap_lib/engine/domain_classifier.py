# mgap_lib/engine/domain_classifier.py — MGAP Library v1.0
"""
Классификатор доменов артефактов HX-AM → UNESCO иерархия.
Работает БЕЗ LLM. Три уровня:

  Tier 1 (exact):    точный словарь "biology" → (1, 1.5, "Молекулярная биология")
  Tier 2 (keyword):  подстрока в домене → (disc_code, sector_code)
  Tier 3 (cosine):   sentence-transformers cosine по имени сектора (all-MiniLM-L6-v2)

Возвращает DomainClassificationResult с уровнем уверенности.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("MGAP.classifier")

_DOMAIN_MAP_PATH = Path(__file__).parent.parent / "data" / "domain_map.json"


@dataclass
class DomainClassificationResult:
    domain_raw:      str
    disc_code:       Optional[str]     = None
    sector_code:     Optional[str]     = None
    specialization:  Optional[str]     = None
    method:          str               = "unknown"  # exact | keyword | cosine | manual | fallback
    confidence:      float             = 1.0
    disc_name_ru:    Optional[str]     = None
    sector_name_ru:  Optional[str]     = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "domain_raw":     self.domain_raw,
            "disc_code":      self.disc_code,
            "sector_code":    self.sector_code,
            "specialization": self.specialization,
            "method":         self.method,
            "confidence":     self.confidence,
            "disc_name_ru":   self.disc_name_ru,
            "sector_name_ru": self.sector_name_ru,
        }


class DomainClassifier:
    """
    Классификатор без LLM.

    Пример:
        clf = DomainClassifier()
        result = clf.classify("ecology")
        print(result.sector_code)  # "4.2"
        print(result.method)       # "exact"
    """

    def __init__(self, domain_map_path: Optional[Path] = None):
        self._map_path  = domain_map_path or _DOMAIN_MAP_PATH
        self._exact:    Dict[str, Dict]  = {}
        self._keywords: Dict[str, Dict]  = {}
        self._sectors:  List[Dict]       = []
        self._disc_names: Dict[str, str] = {}  # disc_code → name_ru
        self._sector_names: Dict[str, str] = {}  # sector_code → name_ru
        self._sector_vecs = None          # lazy: sentence-transformers embeddings
        self._model       = None          # lazy: SentenceTransformer
        self._load_map()

    # ── загрузка карты ──────────────────────────────────────────────────────

    def _load_map(self):
        if not self._map_path.exists():
            logger.warning(f"domain_map.json not found at {self._map_path}")
            return
        try:
            data = json.loads(self._map_path.read_text(encoding="utf-8"))
            self._exact    = data.get("exact", {})
            self._keywords = data.get("keywords", {})
            self._sectors  = data.get("fallback_sector_names", [])
            for s in self._sectors:
                self._sector_names[s["sector_code"]] = s["name_ru"]
            # Disc names from first occurrence
            for s in self._sectors:
                if s["disc_code"] not in self._disc_names:
                    disc_ru = {
                        "1": "Естественные науки",
                        "2": "Инженерные и технологические науки",
                        "3": "Медицинские и медико-биологические науки",
                        "4": "Сельскохозяйственные науки",
                        "5": "Социальные науки",
                        "6": "Гуманитарные науки",
                    }.get(s["disc_code"], "")
                    self._disc_names[s["disc_code"]] = disc_ru
            logger.info(f"DomainClassifier: loaded {len(self._exact)} exact, "
                        f"{len(self._keywords)} keyword, {len(self._sectors)} sectors")
        except Exception as e:
            logger.error(f"DomainClassifier._load_map failed: {e}")

    # ── публичный API ────────────────────────────────────────────────────────

    def classify(self, domain: str) -> DomainClassificationResult:
        """Классифицирует домен. Возвращает DomainClassificationResult."""
        if not domain or domain.strip() == "":
            return DomainClassificationResult(
                domain_raw="", disc_code="1", sector_code="1.1",
                method="fallback", confidence=0.1
            )

        norm = domain.strip().lower()

        # Tier 1: exact
        if norm in self._exact:
            rec = self._exact[norm]
            return self._make_result(norm, rec, method="exact", confidence=1.0)

        # Tier 2: keyword substring
        result = self._tier2_keyword(norm)
        if result:
            return result

        # Tier 3: cosine
        result = self._tier3_cosine(norm)
        if result:
            return result

        # Fallback
        logger.warning(f"DomainClassifier: no match for '{domain}' — fallback to 1.1 Mathematics")
        return DomainClassificationResult(
            domain_raw=domain, disc_code="1", sector_code="1.1",
            specialization=None, method="fallback", confidence=0.0,
            disc_name_ru=self._disc_names.get("1"),
            sector_name_ru=self._sector_names.get("1.1"),
        )

    def classify_batch(self, domains: List[str]) -> Dict[str, DomainClassificationResult]:
        """Классифицирует список доменов. Возвращает словарь domain→result."""
        return {d: self.classify(d) for d in domains}

    # ── Tier 2 ───────────────────────────────────────────────────────────────

    def _tier2_keyword(self, norm: str) -> Optional[DomainClassificationResult]:
        for kw, rec in self._keywords.items():
            if kw in norm:
                logger.debug(f"DomainClassifier Tier2: '{norm}' matched keyword '{kw}'")
                return self._make_result(norm, rec, method="keyword", confidence=0.8)
        return None

    # ── Tier 3 ───────────────────────────────────────────────────────────────

    def _tier3_cosine(self, norm: str) -> Optional[DomainClassificationResult]:
        """Cosine similarity против имён секторов на английском."""
        if not self._sectors:
            return None
        try:
            model = self._get_st_model()
            if model is None:
                return None
            import numpy as np
            if self._sector_vecs is None:
                texts = [s["name_en"] + " " + s["name_ru"] for s in self._sectors]
                self._sector_vecs = model.encode(texts, show_progress_bar=False)
            q_vec = model.encode(norm)
            # cosine similarity
            norms = np.linalg.norm(self._sector_vecs, axis=1) * np.linalg.norm(q_vec)
            norms = np.where(norms == 0, 1e-9, norms)
            sims  = (self._sector_vecs @ q_vec) / norms
            best_i = int(np.argmax(sims))
            best_s = float(sims[best_i])
            if best_s < 0.3:
                return None
            sec = self._sectors[best_i]
            logger.debug(f"DomainClassifier Tier3: '{norm}' → sector '{sec['sector_code']}' sim={best_s:.3f}")
            return self._make_result(
                norm,
                {"disc_code": sec["disc_code"], "sector_code": sec["sector_code"], "specialization": None},
                method="cosine",
                confidence=round(best_s, 3),
            )
        except Exception as e:
            logger.warning(f"DomainClassifier Tier3 failed: {e}")
            return None

    def _get_st_model(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer("all-MiniLM-L6-v2")
            except ImportError:
                logger.warning("sentence-transformers not installed; Tier3 disabled")
        return self._model

    # ── helpers ──────────────────────────────────────────────────────────────

    def _make_result(self, domain_raw: str, rec: Dict, method: str, confidence: float) -> DomainClassificationResult:
        dc = rec.get("disc_code")
        sc = rec.get("sector_code")
        return DomainClassificationResult(
            domain_raw=domain_raw,
            disc_code=dc,
            sector_code=sc,
            specialization=rec.get("specialization"),
            method=method,
            confidence=confidence,
            disc_name_ru=self._disc_names.get(dc) if dc else None,
            sector_name_ru=self._sector_names.get(sc) if sc else None,
        )

    def describe(self, domain: str) -> str:
        """Возвращает читаемую строку классификации для отладки."""
        r = self.classify(domain)
        parts = []
        if r.disc_name_ru:  parts.append(r.disc_name_ru)
        if r.sector_name_ru: parts.append(r.sector_name_ru)
        if r.specialization: parts.append(r.specialization)
        desc = " → ".join(parts) if parts else "Неизвестно"
        return f"[{r.disc_code}/{r.sector_code}] {desc} (method={r.method}, conf={r.confidence})"