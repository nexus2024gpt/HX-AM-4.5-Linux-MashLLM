# mgap_lib/api/dependencies.py — MGAP Library v1.0
"""
FastAPI Dependency Injection для MGAP Library.

Использование в роутах:
    from mgap_lib.api.dependencies import get_engine, get_session

    @router.get("/match/{artifact_id}")
    def match(artifact_id: str, engine: MGAPEngine = Depends(get_engine)):
        return engine.match_artifact(artifact_id)

Синглтоны:
  _engine  — MGAPEngine, создаётся один раз при старте
  _session — SQLAlchemy Session (новая на каждый запрос)
"""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import Generator, Optional

logger = logging.getLogger("MGAP.deps")

# ── Глобальные синглтоны ──────────────────────────────────────────────────────
_engine_instance = None
_session_factory = None


def init_engine(
    registry_path: str = "mgap_registry.json",
    db_url: Optional[str] = None,
    artifacts_dir: str = "artifacts",
    use_llm: bool = True,
    gap_mode: str = "max",
):
    """
    Инициализирует MGAPEngine синглтон.
    Вызывать один раз в lifespan FastAPI приложения (или при старте).

    Если db_url передан — использует SQLAlchemy БД.
    Иначе — JSON-фолбэк из mgap_registry.json.
    """
    global _engine_instance, _session_factory

    from mgap_lib.engine.matcher import MGAPEngine

    if db_url:
        from mgap_lib.models.database import (
            init_default_db, get_session_factory, create_tables
        )
        db_engine    = init_default_db(db_url)
        _session_factory = get_session_factory(db_engine)
        from mgap_lib.engine.registry import RegistryLoader
        session = _session_factory()
        registry = RegistryLoader(db_session=session)
        session.close()
        _engine_instance = MGAPEngine(
            registry=registry,
            artifacts_dir=Path(artifacts_dir),
            gap_mode=gap_mode,
        )
        logger.info(f"MGAPEngine initialized with DB: {db_url}")
    else:
        _engine_instance = MGAPEngine.from_json(
            registry_path=registry_path,
            artifacts_dir=artifacts_dir,
            use_llm=use_llm,
            gap_mode=gap_mode,
        )
        logger.info(f"MGAPEngine initialized from JSON: {registry_path}")

    return _engine_instance


def get_engine():
    """FastAPI Depends: возвращает MGAPEngine синглтон."""
    global _engine_instance
    if _engine_instance is None:
        # Ленивая инициализация с дефолтными настройками
        from mgap_lib.config.settings import settings
        init_engine(
            registry_path=str(settings.registry_path),
            db_url=settings.db_url if settings.db_url != "sqlite:///mgap.db" else None,
            artifacts_dir=str(settings.artifacts_dir),
            use_llm=settings.use_llm,
            gap_mode=settings.gap_mode,
        )
    return _engine_instance


def get_session() -> Generator:
    """
    FastAPI Depends: возвращает SQLAlchemy сессию.
    Только если инициализирована с db_url.
    """
    global _session_factory
    if _session_factory is None:
        yield None
        return
    session = _session_factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def get_classifier():
    """FastAPI Depends: возвращает DomainClassifier из engine."""
    engine = get_engine()
    return engine.classifier


def get_registry():
    """FastAPI Depends: возвращает RegistryLoader из engine."""
    engine = get_engine()
    return engine.registry
