# mgap_lib/models/database.py — MGAP Library v1.0
"""
SQLAlchemy ORM — полная иерархия UNESCO + модели MGAP.

Таблицы:
  disciplines      — 6 крупных разделов UNESCO
  sectors          — ~28 секторов (2-й уровень)
  specializations  — специализации (3-й уровень)
  mgap_models      — отраслевые модели (привязаны к специализации)
  artifact_runs    — история прогонов артефактов через MGAP

Совместимость: SQLite (пилот) и PostgreSQL (prod).
JSON-поля хранятся как Text с сериализацией.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from sqlalchemy import (
    Boolean, ForeignKey, Index, String, Text, create_engine, event
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship, sessionmaker, Session


# ── Base ─────────────────────────────────────────────────────────────────────

class Base(DeclarativeBase):
    pass


# ── JSON helpers ─────────────────────────────────────────────────────────────

def _json_get(obj, attr: str) -> Any:
    raw = getattr(obj, attr, None)
    if raw is None:
        return None
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except Exception:
            return raw
    return raw


def _json_set(obj, attr: str, value: Any):
    if value is None:
        setattr(obj, attr, None)
    elif isinstance(value, str):
        setattr(obj, attr, value)
    else:
        setattr(obj, attr, json.dumps(value, ensure_ascii=False))


# ══════════════════════════════════════════════════════════════════════════════
# ИЕРАРХИЯ UNESCO
# ══════════════════════════════════════════════════════════════════════════════

class Discipline(Base):
    __tablename__ = "disciplines"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    code: Mapped[str] = mapped_column(String(4), nullable=False, unique=True, index=True)
    name_ru: Mapped[str] = mapped_column(String(200), nullable=False)
    name_en: Mapped[str] = mapped_column(String(200), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    sectors: Mapped[List["Sector"]] = relationship(
        "Sector", back_populates="discipline", cascade="all, delete-orphan"
    )

    def __repr__(self):
        return f"<Discipline {self.code} {self.name_ru}>"


class Sector(Base):
    __tablename__ = "sectors"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    code: Mapped[str] = mapped_column(String(8), nullable=False, unique=True, index=True)
    name_ru: Mapped[str] = mapped_column(String(200), nullable=False)
    name_en: Mapped[str] = mapped_column(String(200), nullable=False)
    disc_code: Mapped[str] = mapped_column(String(4), ForeignKey("disciplines.code"), nullable=False)

    discipline: Mapped["Discipline"] = relationship("Discipline", back_populates="sectors")
    specializations: Mapped[List["Specialization"]] = relationship(
        "Specialization", back_populates="sector", cascade="all, delete-orphan"
    )

    def __repr__(self):
        return f"<Sector {self.code} {self.name_ru}>"


class Specialization(Base):
    __tablename__ = "specializations"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(300), nullable=False)
    sector_code: Mapped[str] = mapped_column(String(8), ForeignKey("sectors.code"), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    sector: Mapped["Sector"] = relationship("Sector", back_populates="specializations")
    models: Mapped[List["MGAPModel"]] = relationship("MGAPModel", back_populates="specialization")

    def __repr__(self):
        return f"<Specialization {self.name}>"


# ══════════════════════════════════════════════════════════════════════════════
# МОДЕЛИ MGAP
# ══════════════════════════════════════════════════════════════════════════════

class MGAPModel(Base):
    __tablename__ = "mgap_models"

    id: Mapped[str] = mapped_column(String(20), primary_key=True)
    name: Mapped[str] = mapped_column(String(300), nullable=False)
    logia: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    industry: Mapped[Optional[str]] = mapped_column(String(200), nullable=True)
    math_type: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    disc_code: Mapped[Optional[str]] = mapped_column(String(4), nullable=True, index=True)
    sector_code: Mapped[Optional[str]] = mapped_column(String(8), nullable=True, index=True)
    specialization_id: Mapped[Optional[int]] = mapped_column(ForeignKey("specializations.id"), nullable=True)

    # JSON-поля хранятся как строки
    _programs: Mapped[Optional[str]] = mapped_column("programs", Text, nullable=True)
    _four_d_matrix: Mapped[Optional[str]] = mapped_column("four_d_matrix", Text, nullable=True)
    _expected_ranges: Mapped[Optional[str]] = mapped_column("expected_ranges", Text, nullable=True)
    _weights: Mapped[Optional[str]] = mapped_column("weights", Text, nullable=True)
    _critical_thresholds: Mapped[Optional[str]] = mapped_column("critical_thresholds", Text, nullable=True)
    _translation_map: Mapped[Optional[str]] = mapped_column("translation_map", Text, nullable=True)
    _example_data: Mapped[Optional[str]] = mapped_column("example_data", Text, nullable=True)

    blind_spot_template: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    math_adaptation_formula: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        default=lambda: datetime.now(timezone.utc)
    )
    updated_at: Mapped[datetime] = mapped_column(
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc)
    )
    is_active: Mapped[bool] = mapped_column(default=True)

    specialization: Mapped[Optional["Specialization"]] = relationship("Specialization", back_populates="models")
    runs: Mapped[List["ArtifactRun"]] = relationship("ArtifactRun", back_populates="model")

    # ── JSON property accessors ───────────────────────────────────────────────

    @property
    def programs(self) -> List[str]:
        v = _json_get(self, "_programs")
        return v if isinstance(v, list) else []

    @programs.setter
    def programs(self, v: List[str]):
        _json_set(self, "_programs", v)

    @property
    def four_d_matrix(self) -> Optional[Dict]:
        return _json_get(self, "_four_d_matrix")

    @four_d_matrix.setter
    def four_d_matrix(self, v):
        _json_set(self, "_four_d_matrix", v)

    @property
    def expected_ranges(self) -> Dict:
        return _json_get(self, "_expected_ranges") or {}

    @expected_ranges.setter
    def expected_ranges(self, v):
        _json_set(self, "_expected_ranges", v)

    @property
    def weights(self) -> Dict:
        return _json_get(self, "_weights") or {}

    @weights.setter
    def weights(self, v):
        _json_set(self, "_weights", v)

    @property
    def critical_thresholds(self) -> Dict:
        return _json_get(self, "_critical_thresholds") or {}

    @critical_thresholds.setter
    def critical_thresholds(self, v):
        _json_set(self, "_critical_thresholds", v)

    @property
    def translation_map(self) -> Dict:
        return _json_get(self, "_translation_map") or {}

    @translation_map.setter
    def translation_map(self, v):
        _json_set(self, "_translation_map", v)

    @property
    def example_data(self) -> Optional[Dict]:
        return _json_get(self, "_example_data")

    @example_data.setter
    def example_data(self, v):
        _json_set(self, "_example_data", v)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "logia": self.logia,
            "industry": self.industry,
            "math_type": self.math_type,
            "disc_code": self.disc_code,
            "sector_code": self.sector_code,
            "programs": self.programs,
            "four_d_matrix": self.four_d_matrix,
            "expected_ranges": self.expected_ranges,
            "weights": self.weights,
            "critical_thresholds": self.critical_thresholds,
            "translation_map": self.translation_map,
            "example_data": self.example_data,
            "blind_spot_template": self.blind_spot_template,
            "math_adaptation_formula": self.math_adaptation_formula,
            "description": self.description,
        }

    def __repr__(self):
        return f"<MGAPModel {self.id} '{self.name}' [{self.math_type}]>"


# ══════════════════════════════════════════════════════════════════════════════
# ИСТОРИЯ ПРОГОНОВ
# ══════════════════════════════════════════════════════════════════════════════

class ArtifactRun(Base):
    __tablename__ = "artifact_runs"
    __table_args__ = (
        Index("ix_artifact_runs_artifact_id", "artifact_id"),
        Index("ix_artifact_runs_model_id", "model_id"),
        Index("ix_artifact_runs_created_at", "created_at"),
    )

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    artifact_id: Mapped[str] = mapped_column(String(50), nullable=False)
    model_id: Mapped[str] = mapped_column(String(20), ForeignKey("mgap_models.id"), nullable=False)
    domain: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    disc_code: Mapped[Optional[str]] = mapped_column(String(4), nullable=True)
    sector_code: Mapped[Optional[str]] = mapped_column(String(8), nullable=True)
    resonance: Mapped[Optional[float]] = mapped_column(nullable=True)
    risk_level: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)
    _results: Mapped[Optional[str]] = mapped_column("results", Text, nullable=True)
    status: Mapped[str] = mapped_column(String(20), default="ok")
    error_msg: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(default=lambda: datetime.now(timezone.utc))

    model: Mapped["MGAPModel"] = relationship("MGAPModel", back_populates="runs")

    @property
    def results(self) -> Optional[Dict]:
        return _json_get(self, "_results")

    @results.setter
    def results(self, v):
        _json_set(self, "_results", v)

    def __repr__(self):
        return f"<ArtifactRun art={self.artifact_id} model={self.model_id} resonance={self.resonance}>"


# ══════════════════════════════════════════════════════════════════════════════
# ФАБРИКА СЕССИЙ
# ══════════════════════════════════════════════════════════════════════════════

def create_db_engine(db_url: str = "sqlite:///mgap.db"):
    connect_args = {"check_same_thread": False} if db_url.startswith("sqlite") else {}
    engine = create_engine(db_url, connect_args=connect_args, echo=False)

    if db_url.startswith("sqlite"):
        @event.listens_for(engine, "connect")
        def set_wal(dbapi_conn, conn_record):
            cursor = dbapi_conn.cursor()
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.close()
    return engine


def create_tables(engine):
    Base.metadata.create_all(engine)


def get_session_factory(engine) -> sessionmaker:
    return sessionmaker(bind=engine, expire_on_commit=False)


_default_engine = None
_default_factory = None


def init_default_db(db_url: str = "sqlite:///mgap.db"):
    global _default_engine, _default_factory
    _default_engine = create_db_engine(db_url)
    create_tables(_default_engine)
    _default_factory = get_session_factory(_default_engine)
    return _default_engine


def get_session() -> Session:
    if _default_factory is None:
        init_default_db()
    return _default_factory()
