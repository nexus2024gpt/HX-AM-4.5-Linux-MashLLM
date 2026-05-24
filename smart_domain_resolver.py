# smart_domain_resolver.py — HX-AM v4.6
"""
Трёхуровневый определитель домена без LLM.

Уровень 1: Алиасы (мгновенно, точно)
Уровень 2: Ключевые слова (быстро, покрывает ~80% случаев)
Уровень 3: Семантика по архиву (через SemanticSpace.nearest)

LLM вызывается ТОЛЬКО если все три уровня дают confidence < 0.5.
Это экономит ~2 вызова LLM на каждый запрос.
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING, Dict, Optional, Tuple

from domain_config import (
    DOMAIN_ALIASES, DOMAIN_KEYWORDS, VALID_DOMAINS, NEW_DOMAIN_PARAMS,
)

if TYPE_CHECKING:
    from invariant_engine import SemanticSpace

logger = logging.getLogger("HXAM.domain")


class SmartDomainResolver:

    def __init__(self, space: Optional["SemanticSpace"] = None):
        self._space = space

    def resolve(
        self,
        text: str,
        llm_domain: Optional[str] = None,
        hypothesis: str = "",
        mechanism: str = "",
    ) -> Tuple[str, float, str]:
        """
        Определяет домен. Возвращает (domain, confidence, method).

        Args:
            text:       LLM-ответ с доменом (может быть "general" или пустым)
            llm_domain: уже извлечённый домен из LLM (если есть)
            hypothesis: текст гипотезы для семантического поиска
            mechanism:  текст механизма

        Returns:
            (domain, confidence 0-1, method)
        """
        full_text = f"{hypothesis} {mechanism} {text}".strip()

        # ── Уровень 1: Алиасы ─────────────────────────────────────────────
        candidate = (llm_domain or text or "").strip().lower()
        if candidate in DOMAIN_ALIASES:
            resolved = DOMAIN_ALIASES[candidate]
            if resolved != "general":
                return resolved, 1.0, "alias"
        if candidate in VALID_DOMAINS and candidate != "general":
            return candidate, 1.0, "exact"

        # ── Уровень 2: Ключевые слова ──────────────────────────────────────
        kw_result = self._keyword_match(full_text)
        if kw_result and kw_result[1] >= 0.6:
            return kw_result[0], kw_result[1], "keyword"

        # ── Уровень 3: Семантика по архиву ────────────────────────────────
        if self._space and len(self._space.vectors) >= 5 and hypothesis:
            sem_result = self._semantic_match(hypothesis + " " + mechanism)
            if sem_result and sem_result[1] >= 0.6:
                # Проверяем что кандидат не general
                if sem_result[0] != "general":
                    logger.info(
                        f"SmartDomainResolver: semantic match "
                        f"'{sem_result[0]}' conf={sem_result[1]:.2f}"
                    )
                    return sem_result[0], sem_result[1], "semantic"

        # ── Если LLM дал конкретный домен (не general) — доверяем ─────────
        if kw_result and kw_result[1] >= 0.4:
            return kw_result[0], kw_result[1], "keyword_weak"

        # ── Возвращаем general только как последний вариант ────────────────
        logger.warning(
            f"SmartDomainResolver: could not classify, using 'general'. "
            f"Text[:80]: {full_text[:80]}"
        )
        return "general", 0.1, "fallback"

    def _keyword_match(self, text: str) -> Optional[Tuple[str, float]]:
        """Подсчёт ключевых слов по домену. Возвращает лучший домен."""
        text_lower = text.lower()
        scores: Dict[str, int] = {}

        for domain, keywords in DOMAIN_KEYWORDS.items():
            count = sum(1 for kw in keywords if kw in text_lower)
            if count > 0:
                scores[domain] = count

        if not scores:
            return None

        best_domain = max(scores, key=lambda d: scores[d])
        total_keywords = len(DOMAIN_KEYWORDS[best_domain])
        # confidence: нелинейная — 1 совпадение даёт 0.5, 3+ → 0.85+
        raw_conf = scores[best_domain] / max(total_keywords * 0.3, 1)
        confidence = round(min(0.95, 0.4 + raw_conf * 0.55), 2)

        return best_domain, confidence

    def _semantic_match(self, query: str) -> Optional[Tuple[str, float]]:
        """
        Ищет ближайшие артефакты в архиве и берёт домен большинства.
        Работает ТОЛЬКО если в архиве есть хотя бы 5 артефактов.
        Важно: исключает "general" артефакты из голосования.
        """
        if not self._space:
            return None

        similar = self._space.nearest(query, top_k=7, threshold=0.50)
        # Фильтруем general — они не должны влиять на определение
        valid = [s for s in similar if s.get("domain", "general") != "general"]

        if not valid:
            return None

        # Взвешенное голосование по similarity
        votes: Dict[str, float] = {}
        for s in valid:
            d = s.get("domain", "general")
            votes[d] = votes.get(d, 0) + s["similarity"]

        best = max(votes, key=lambda d: votes[d])
        # Нормализуем в [0, 1]
        total_weight = sum(votes.values())
        confidence = round(votes[best] / max(total_weight, 1e-9), 2)

        return best, confidence