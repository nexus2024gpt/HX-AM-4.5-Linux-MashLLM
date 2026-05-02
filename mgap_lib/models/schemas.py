# mgap_lib/models/schemas.py — MGAP Library v1.0
"""
Pydantic-схемы для API-слоя и CLI.
Все поля опциональны где возможно — для совместимости с частичными данными данными HX-AM.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


# ── UNESCO Hierarchy ─────────────────────────────────────────────────────────

class SpecializationSchema(BaseModel):
    name: str
    description: Optional[str] = None


class SectorSchema(BaseModel):
    code: str
    name_ru: str
    name_en: str
    specializations: List[str] = Field(default_factory=list)


class DisciplineSchema(BaseModel):
    code: str
    name_ru: str
    name_en: str
    sectors: List[SectorSchema] = Field(default_factory=list)


# ── Model Schemas ─────────────────────────────────────────────────────────────

class CriticalThresholds(BaseModel):
    eta_max: Optional[float] = None
    tau_max: Optional[float] = None
    K_min:   Optional[float] = None


class ModelCreateSchema(BaseModel):
    """Схема для создания новой отраслевой модели через API."""
    model_id:                 str
    name:                     str
    logia:                    Optional[str] = None
    industry:                 Optional[str] = None
    math_type:                str           = Field(..., pattern="^(kuramoto|percolation|ising|delay|lotka_volterra|graph_invariant)$")
    disc_code:                Optional[str] = None
    sector_code:              Optional[str] = None
    specialization_name:      Optional[str] = None
    programs:                 List[str]     = Field(default_factory=list)
    four_d_matrix:            Optional[Dict[str, Any]] = None
    expected_ranges:          Optional[Dict[str, Any]] = None
    weights:                  Optional[Dict[str, Any]] = None
    critical_thresholds:      Optional[Dict[str, Any]] = None
    translation_map:          Optional[Dict[str, Any]] = None
    example_data:             Optional[Dict[str, Any]] = None
    blind_spot_template:      Optional[str] = None
    math_adaptation_formula:  Optional[str] = None
    description:              Optional[str] = None


class ModelSummarySchema(BaseModel):
    """Краткое представление модели для списков."""
    model_id:    str
    name:        str
    logia:       Optional[str]
    industry:    Optional[str]
    math_type:   str
    disc_code:   Optional[str]
    sector_code: Optional[str]
    programs:    List[str]


# ── Artifact Schemas ──────────────────────────────────────────────────────────

class ArtifactInput(BaseModel):
    """Минимальный ввод артефакта для сопоставления."""
    id:   Optional[str] = None
    data: Optional[Dict[str, Any]] = None
    simulation: Optional[Dict[str, Any]] = None


class DomainClassificationResult(BaseModel):
    """Результат классификации домена артефакта."""
    domain_raw:      str
    disc_code:       Optional[str]
    sector_code:     Optional[str]
    specialization:  Optional[str]
    method:          str   # exact | keyword | cosine | manual
    confidence:      float = 1.0


# ── Match Schemas ─────────────────────────────────────────────────────────────

class GapComponents(BaseModel):
    """Компоненты разрыва между артефактом и моделью."""
    eta_gap: float = 0.0
    tau_gap: float = 0.0
    K_gap:   float = 0.0
    composite: float = 0.0   # max(eta_gap, tau_gap, K_gap)


class ThresholdResult(BaseModel):
    eta_critical:    float
    tau_robustness:  float
    lyapunov_max:    Optional[float] = None
    stability_score: Optional[float] = None
    survival_verified: Optional[bool] = None


class MatchResult(BaseModel):
    """Полный результат сопоставления одного артефакта с одной моделью."""
    artifact_id:     str
    model_id:        str
    model_name:      str
    logia:           Optional[str]
    industry:        Optional[str]
    disc_code:       Optional[str]
    sector_code:     Optional[str]
    programs:        List[str]
    resonance:       float
    math_type_match: bool
    gap:             GapComponents
    thresholds:      ThresholdResult
    translation:     Dict[str, Any]
    blind_spot:      str
    adaptation:      Dict[str, Any]
    calculation:     Dict[str, Any]
    verdict:         Dict[str, Any]
    artifact_summary: Dict[str, Any]
    generated_at:    str


# ── API Request / Response ────────────────────────────────────────────────────

class MatchRequest(BaseModel):
    artifact_id:     Optional[str] = None
    artifact_json:   Optional[Dict[str, Any]] = None
    model_filter:    Optional[List[str]] = None   # ограничить по model_id
    sector_filter:   Optional[str] = None          # ограничить пп sector_code
    math_type_only:  bool = True
    top_k:           int  = Field(default=3, ge=1, le=20)


class MatchResponse(BaseModel):
    artifact_id: str
    matches:     List[MatchResult]
    total:       int
    domain_classification: Optional[DomainClassificationResult] = None


class EvaluateAllRequest(BaseModel):
    artifact_ids:   List[str]
    top_k:          int = 3
    math_type_only: bool = True
    save_to_db:     bool = True


class EvaluateAllResponse(BaseModel):
    processed:  int
    errors:     int
    results:    Dict[str, List[MatchResult]]
