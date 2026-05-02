# mgap_lib/api/routes.py — MGAP Library v1.0
"""
FastAPI роутер MGAP Library.

Эндпоинты:
  GET  /mgap/match/{artifact_id}     — топ-K моделей для артефакта
  POST /mgap/match                   — матч с передачей JSON артефакта
  GET  /mgap/batch                   — прогон всех артефактов
  GET  /mgap/registry                — список всех моделей
  GET  /mgap/model/{model_id}        — конкретная модель
  GET  /mgap/runs                    — история прогонов из БД
  GET  /mgap/classify                — классификация домена → UNESCO
  GET  /mgap/taxonomy                — UNESCO дерево
  GET  /mgap/stats                   — статистика

Интеграция в hxam_v_4_server.py:
    from mgap_lib.api.routes import mgap_router
    from mgap_lib.api.dependencies import init_engine
    init_engine()
    app.include_router(mgap_router)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from mgap_lib.api.dependencies import get_engine, get_classifier, get_registry

logger = logging.getLogger("MGAP.api")

mgap_router = APIRouter(prefix="/mgap", tags=["MGAP"])


# ════════════════════════════════════════════════════════════════
# PYDANTIC REQUEST MODELS
# ════════════════════════════════════════════════════════════════

class MatchByJsonRequest(BaseModel):
    """Передаём артефакт целиком (не читая с диска)."""
    artifact_id:   Optional[str]              = None
    artifact_json: Dict[str, Any]
    top_k:         int                         = 3
    math_type_only: bool                       = True
    sector_filter: Optional[str]               = None
    save_to_db:    bool                        = False


class BatchRequest(BaseModel):
    top_k:           int   = 2
    math_type_only:  bool  = True
    min_resonance:   float = 0.3
    save_to_db:      bool  = False


# ════════════════════════════════════════════════════════════════
# ОСНОВНЫЕ ЭНДПОИНТЫ
# ════════════════════════════════════════════════════════════════

@mgap_router.get("/match/{artifact_id}",
                 summary="Найти отраслевые модели для артефакта")
def match_artifact(
    artifact_id:   str,
    top_k:         int  = Query(default=3, ge=1, le=20),
    all_types:     bool = Query(default=False),
    model_id:      str  = Query(default=""),
    sector_filter: str  = Query(default=""),
    engine=Depends(get_engine),
):
    """
    Ищет топ-K отраслевых моделей для артефакта по artifact_id.

    - **artifact_id**: ID артефакта HX-AM (без .json)
    - **top_k**: количество лучших совпадений (1–20)
    - **all_types**: если true — матч по всем math_type, иначе только совпадающие
    - **model_id**: ограничить поиск конкретной моделью
    - **sector_filter**: ограничить по сектору UNESCO (напр. "5.1")
    """
    try:
        results = engine.match_artifact(
            artifact_id    = artifact_id,
            top_k          = top_k,
            math_type_only = not all_types,
            model_id       = model_id or None,
            sector_filter  = sector_filter or None,
        )
        return {
            "artifact_id": artifact_id,
            "matches":     results,
            "total":       len(results),
        }
    except Exception as e:
        logger.error(f"match_artifact error: {e}", exc_info=True)
        raise HTTPException(500, str(e))


@mgap_router.post("/match",
                  summary="Матч с передачей JSON артефакта")
def match_by_json(req: MatchByJsonRequest, engine=Depends(get_engine)):
    """
    Сопоставляет артефакт, переданный в теле запроса.
    Полезно для тестирования без сохранённого файла.
    """
    if not req.artifact_id and not req.artifact_json.get("id"):
        raise HTTPException(400, "artifact_id or artifact_json.id required")

    art_id = req.artifact_id or req.artifact_json.get("id", "inline")

    # Временно пишем в память — engine._load_artifact смотрит на диск,
    # поэтому патчим через inline метод
    try:
        four_d = req.artifact_json.get("data", {}).get("gen", {}).get("four_d_matrix")
        if not four_d:
            four_d = req.artifact_json.get("four_d_matrix")
        if not four_d:
            raise HTTPException(400, "No four_d_matrix in artifact_json")

        from mgap_lib.engine.matcher import _flat_4d, _norm_mt, _compute_resonance, _art_vector, _extract_thresholds, _generate_code, _calculate_example
        from mgap_lib.engine.domain_classifier import DomainClassifier

        flat     = _flat_4d(four_d)
        art_math = _norm_mt(flat["model"])
        art_vec  = _art_vector(four_d)
        domain   = req.artifact_json.get("data", {}).get("domain", "general") or "general"
        domain_cls = DomainClassifier().classify(domain)

        candidates = engine.registry.get_all()
        if req.model_id: candidates = [m for m in candidates if m.get("id") == req.model_id]
        if req.math_type_only:
            candidates = [m for m in candidates if _norm_mt(m.get("math_type", "")) == art_math]
        if req.sector_filter:
            candidates = [m for m in candidates if m.get("sector_code") == req.sector_filter]

        scored = sorted(
            [(_compute_resonance(art_vec, m, art_math), m) for m in candidates],
            key=lambda x: -x[0]
        )

        results = [
            engine._build_match(
                artifact_id=art_id,
                artifact=req.artifact_json,
                four_d=four_d,
                flat=flat,
                thresholds=_extract_thresholds(req.artifact_json, m),
                art_math=art_math,
                model=m,
                resonance=r,
                domain_cls=domain_cls,
            )
            for r, m in scored[:req.top_k]
        ]

        return {"artifact_id": art_id, "matches": results, "total": len(results)}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"match_by_json error: {e}", exc_info=True)
        raise HTTPException(500, str(e))


@mgap_router.post("/batch",
                  summary="Batch: все артефакты × все модели")
def batch_match(req: BatchRequest = BatchRequest(), engine=Depends(get_engine)):
    """
    Прогоняет все артефакты из artifacts/ через MGAPEngine.
    Фильтрует по min_resonance. Возвращает словарь {artifact_id: [matches]}.
    """
    try:
        results = engine.match_batch(
            top_k          = req.top_k,
            math_type_only = req.math_type_only,
            min_resonance  = req.min_resonance,
            save_to_db     = req.save_to_db,
        )
        return {
            "results":         results,
            "artifacts_count": len(results),
            "models_count":    engine.registry.count(),
        }
    except Exception as e:
        raise HTTPException(500, str(e))


# ════════════════════════════════════════════════════════════════
# РЕЕСТР МОДЕЛЕЙ
# ════════════════════════════════════════════════════════════════

@mgap_router.get("/registry",
                 summary="Список всех моделей реестра")
def get_registry_list(
    math_type:    Optional[str] = Query(default=None),
    sector_code:  Optional[str] = Query(default=None),
    registry=Depends(get_registry),
):
    """
    Возвращает список моделей с фильтрацией.
    - **math_type**: фильтр по типу модели (kuramoto/percolation/delay/...)
    - **sector_code**: фильтр по сектору UNESCO (1.5/5.1/...)
    """
    models = registry.get_all()
    if math_type:
        from mgap_lib.engine.matcher import _norm_mt
        models = [m for m in models if _norm_mt(m.get("math_type", "")) == _norm_mt(math_type)]
    if sector_code:
        models = [m for m in models if m.get("sector_code") == sector_code]
    summary = [
        {
            "id":          m.get("id"),
            "name":        m.get("name"),
            "logia":       m.get("logia"),
            "industry":    m.get("industry"),
            "math_type":   m.get("math_type"),
            "disc_code":   m.get("disc_code"),
            "sector_code": m.get("sector_code"),
            "programs":    m.get("programs", []),
        }
        for m in models
    ]
    return {"models": summary, "total": len(summary), "version": "v1.0"}


@mgap_router.get("/model/{model_id}",
                 summary="Полные данные одной модели")
def get_model(model_id: str, registry=Depends(get_registry)):
    """Возвращает полный словарь модели по ID."""
    model = registry.get_by_id(model_id)
    if not model:
        raise HTTPException(404, f"Model '{model_id}' not found")
    return model


# ════════════════════════════════════════════════════════════════
# КЛАССИФИКАЦИЯ ДОМЕНОВ
# ════════════════════════════════════════════════════════════════

@mgap_router.get("/classify",
                 summary="Классификация домена → UNESCO")
def classify_domain(
    domain: str = Query(..., description="Домен на английском: biology, economics, etc."),
    classifier=Depends(get_classifier),
):
    """
    Классифицирует домен по UNESCO-иерархии.
    Возвращает disc_code, sector_code, specialization, метод и уверенность.
    """
    result = classifier.classify(domain)
    desc   = classifier.describe(domain)
    return {
        "domain":       domain,
        "classification": result.to_dict(),
        "description":  desc,
    }


@mgap_router.get("/classify/batch",
                 summary="Batch-классификация доменов")
def classify_domains_batch(
    domains: str = Query(..., description="Домены через запятую: biology,economics,physics"),
    classifier=Depends(get_classifier),
):
    domain_list = [d.strip() for d in domains.split(",") if d.strip()]
    results     = classifier.classify_batch(domain_list)
    return {
        "classifications": {d: r.to_dict() for d, r in results.items()},
        "total": len(results),
    }


# ════════════════════════════════════════════════════════════════
# UNESCO ТАКСОНОМИЯ
# ════════════════════════════════════════════════════════════════

@mgap_router.get("/taxonomy",
                 summary="UNESCO таксономия")
def get_taxonomy(disc_code: Optional[str] = Query(default=None)):
    """
    Возвращает UNESCO таксономию (дисциплины → сектора → специализации).
    Если disc_code задан — возвращает только этот раздел.
    """
    tax_path = Path(__file__).parent.parent / "data" / "unesco_taxonomy.json"
    if not tax_path.exists():
        raise HTTPException(404, "UNESCO taxonomy file not found")
    try:
        taxonomy = json.loads(tax_path.read_text(encoding="utf-8"))
        if disc_code:
            discs = [d for d in taxonomy.get("disciplines", []) if d.get("code") == disc_code]
            if not discs:
                raise HTTPException(404, f"Discipline '{disc_code}' not found")
            return {"discipline": discs[0]}
        return taxonomy
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, str(e))


# ════════════════════════════════════════════════════════════════
# ИСТОРИЯ ПРОГОНОВ (из БД)
# ════════════════════════════════════════════════════════════════

@mgap_router.get("/runs",
                 summary="История прогонов артефактов")
def get_runs(
    artifact_id: Optional[str] = Query(default=None),
    model_id:    Optional[str] = Query(default=None),
    limit:       int           = Query(default=20, ge=1, le=100),
):
    """История матчей из БД. Работает только если БД инициализирована."""
    try:
        from mgap_lib.models.database import ArtifactRun, get_session
        session = get_session()
        q = session.query(ArtifactRun).order_by(ArtifactRun.created_at.desc())
        if artifact_id:
            q = q.filter(ArtifactRun.artifact_id == artifact_id)
        if model_id:
            q = q.filter(ArtifactRun.model_id == model_id)
        runs = q.limit(limit).all()
        session.close()
        return {
            "runs": [
                {
                    "id":          r.id,
                    "artifact_id": r.artifact_id,
                    "model_id":    r.model_id,
                    "domain":      r.domain,
                    "resonance":   r.resonance,
                    "risk_level":  r.risk_level,
                    "status":      r.status,
                    "created_at":  r.created_at.isoformat() if r.created_at else None,
                }
                for r in runs
            ],
            "total": len(runs),
        }
    except ImportError:
        return {"runs": [], "total": 0, "note": "SQLAlchemy not available"}
    except Exception as e:
        logger.warning(f"get_runs: DB not available — {e}")
        return {"runs": [], "total": 0, "note": str(e)}


# ════════════════════════════════════════════════════════════════
# СТАТИСТИКА
# ════════════════════════════════════════════════════════════════

@mgap_router.get("/stats",
                 summary="Статистика MGAP Library")
def get_stats(
    engine=Depends(get_engine),
    registry=Depends(get_registry),
):
    """Сводная статистика: размер реестра, кэш, доступность БД."""
    stats = {
        "registry": {
            "total_models":    registry.count(),
            "math_types":      {},
            "sectors":         {},
        },
        "db_available": False,
        "results_dir_count": 0,
    }

    for m in registry.get_all():
        mt = m.get("math_type", "unknown")
        sc = m.get("sector_code") or "—"
        stats["registry"]["math_types"][mt] = stats["registry"]["math_types"].get(mt, 0) + 1
        stats["registry"]["sectors"][sc]    = stats["registry"]["sectors"].get(sc, 0) + 1

    # Проверка БД
    try:
        from mgap_lib.models.database import get_session, ArtifactRun
        session = get_session()
        count   = session.query(ArtifactRun).count()
        session.close()
        stats["db_available"]   = True
        stats["db_runs_total"]  = count
    except Exception:
        pass

    # Папка результатов
    rdir = engine.results_dir
    if rdir.exists():
        stats["results_dir_count"] = len(list(rdir.glob("*.json")))

    return stats
