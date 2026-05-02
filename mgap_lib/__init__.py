# mgap_lib/__init__.py — MGAP Library v1.0
"""
MGAP Library — Metric GAP переноса инвариантов HX-AM в отраслевые системы.

Быстрый старт:
    from mgap_lib.engine.matcher import MGAPEngine
    engine = MGAPEngine.from_json("mgap_registry.json")
    results = engine.match_artifact("32d4aa917ac4", top_k=3)

Или с БД:
    from mgap_lib.models.database import init_default_db
    from mgap_lib.engine.matcher import MGAPEngine
    init_default_db("sqlite:///mgap.db")
    engine = MGAPEngine()
    results = engine.match_artifact("32d4aa917ac4")
"""

__version__ = "1.0.0"
__author__  = "HX-AM Project"
