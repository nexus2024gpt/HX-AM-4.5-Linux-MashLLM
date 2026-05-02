# mgap_lib/engine/registry.py — MGAP Library v1.0
"""
RegistryLoader — загружает модели MGAP из двух источников:

  1. SQLite/PostgreSQL БД (приоритет) — через MGAPModel ORM
  2. JSON-файл mgap_registry.json (фолбэк) — полная совместимость с v4.4

Автоматически определяет доступный источник.
Кэширует результат в памяти на время сессии.

Пример:
    registry = RegistryLoader()
    models = registry.get_all()
    m = registry.get_by_id("M1")
    by_math = registry.get_by_math_type("kuramoto")
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("MGAP.registry")


class RegistryLoader:
    """
    Загрузчик реестра MGAP-моделей.

    Источники (в порядке приоритета):
      1. SQLAlchemy session (если передана)
      2. JSON-файл mgap_registry.json
    """

    def __init__(
        self,
        registry_path: Optional[Path] = None,
        db_session=None,
    ):
        self._json_path = registry_path or Path("mgap_registry.json")
        self._session   = db_session
        self._cache: Optional[List[Dict]] = None   # кэш загруженных моделей

    # ── публичный API ─────────────────────────────────────────────────────────

    def get_all(self) -> List[Dict]:
        """Возвращает все активные модели."""
        if self._cache is None:
            self._cache = self._load()
        return self._cache

    def get_by_id(self, model_id: str) -> Optional[Dict]:
        """Возвращает модель по ID."""
        for m in self.get_all():
            if m.get("id") == model_id:
                return m
        return None

    def get_by_math_type(self, math_type: str) -> List[Dict]:
        """Возвращает модели с заданным math_type."""
        norm = math_type.lower().strip()
        aliases = {"delay_ode": "delay", "delay-ode": "delay", "graph-invariant": "graph_invariant"}
        norm = aliases.get(norm, norm)
        return [m for m in self.get_all()
                if aliases.get(m.get("math_type","").lower(), m.get("math_type","").lower()) == norm]

    def get_by_sector(self, sector_code: str) -> List[Dict]:
        """Возвращает модели привязанные к сектору UNESCO."""
        return [m for m in self.get_all() if m.get("sector_code") == sector_code]

    def get_summary(self) -> List[Dict]:
        """Краткая сводка для API /registry."""
        return [
            {
                "id":        m.get("id"),
                "name":      m.get("name"),
                "logia":     m.get("logia"),
                "industry":  m.get("industry"),
                "math_type": m.get("math_type"),
                "disc_code": m.get("disc_code"),
                "sector_code": m.get("sector_code"),
                "programs":  m.get("programs", []),
            }
            for m in self.get_all()
        ]

    def invalidate_cache(self):
        """Сбрасывает кэш — перезагрузит модели при следующем запросе."""
        self._cache = None

    def count(self) -> int:
        return len(self.get_all())

    # ── внутренние методы ─────────────────────────────────────────────────────

    def _load(self) -> List[Dict]:
        if self._session is not None:
            return self._load_from_db()
        return self._load_from_json()

    def _load_from_db(self) -> List[Dict]:
        """Загружает модели из SQLAlchemy сессии."""
        try:
            from mgap_lib.models.database import MGAPModel
            models = self._session.query(MGAPModel).filter_by(is_active=True).all()
            result = [m.to_dict() for m in models]
            logger.info(f"RegistryLoader: loaded {len(result)} models from DB")
            return result
        except Exception as e:
            logger.warning(f"RegistryLoader: DB load failed ({e}), falling back to JSON")
            return self._load_from_json()

    def _load_from_json(self) -> List[Dict]:
        """Загружает модели из mgap_registry.json."""
        if not self._json_path.exists():
            logger.error(f"RegistryLoader: {self._json_path} not found")
            return []
        try:
            data = json.loads(self._json_path.read_text(encoding="utf-8"))
            models = data.get("models", [])
            logger.info(f"RegistryLoader: loaded {len(models)} models from {self._json_path}")
            return models
        except Exception as e:
            logger.error(f"RegistryLoader._load_from_json failed: {e}")
            return []

    def upsert_from_json(self, session, registry_path: Optional[Path] = None) -> int:
        """
        Синхронизирует JSON-реестр с БД.
        Создаёт или обновляет MGAPModel записи.
        Возвращает количество обработанных моделей.
        """
        path = registry_path or self._json_path
        if not path.exists():
            raise FileNotFoundError(f"Registry JSON not found: {path}")

        from mgap_lib.models.database import MGAPModel
        data   = json.loads(path.read_text(encoding="utf-8"))
        models = data.get("models", [])
        count  = 0

        for m_data in models:
            m_id = m_data.get("id")
            if not m_id:
                continue
            existing = session.query(MGAPModel).filter_by(id=m_id).first()
            if existing is None:
                existing = MGAPModel(id=m_id)
                session.add(existing)

            existing.name                   = m_data.get("name", "")
            existing.logia                  = m_data.get("logia")
            existing.industry               = m_data.get("industry")
            existing.math_type              = m_data.get("math_type", "kuramoto")
            existing.description            = m_data.get("description")
            existing.programs               = m_data.get("programs", [])
            existing.four_d_matrix          = m_data.get("four_d_matrix")
            existing.expected_ranges        = m_data.get("expected_ranges")
            existing.weights                = m_data.get("weights")
            existing.critical_thresholds    = m_data.get("critical_thresholds")
            existing.translation_map        = m_data.get("translation_map")
            existing.example_data           = m_data.get("example_data")
            existing.blind_spot_template    = m_data.get("blind_spot_template")
            existing.math_adaptation_formula= m_data.get("math_adaptation_formula")
            existing.disc_code              = m_data.get("disc_code")
            existing.sector_code            = m_data.get("sector_code")
            count += 1

        session.commit()
        self.invalidate_cache()
        logger.info(f"RegistryLoader.upsert_from_json: synced {count} models → DB")
        return count
